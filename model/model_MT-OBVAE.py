import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import sys
from tqdm import tqdm
import json
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from module.metrics import evaluate_multitask_predictions, format_results_table
from module.phase_space import generate_phase_space
from module.sensitivity_profile import visualize_decoder_sensitivities
from module.training_monitor import TrainingMonitor
from module.uncertainty_viz import visualize_uncertainty_cases

# --- 项目路径设置 ---
# 假设您的代码文件在 model 文件夹下，而 module 文件夹与 model 在同一级目录
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 1. 数据加载模块
def load_data(train_path, test_path):
    """
    从CSV文件加载并预处理训练和测试数据。
    返回: 训练/测试集的特征和标签, 以及特征名称列表。
    """
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到 - {e}")
        print("请确保您的数据文件路径正确，相对于代码文件位置。")
        sys.exit(1)

    feature_names = train_data.columns[:-3].tolist()
    X_train = train_data.iloc[:, :-3].values.astype(np.float32)
    y_train = train_data.iloc[:, -3:].values.astype(np.int64)
    X_test = test_data.iloc[:, :-3].values.astype(np.float32)
    y_test = test_data.iloc[:, -3:].values.astype(np.int64)
    return X_train, y_train, X_test, y_test, feature_names


# 2. MT-OBVAE 模型定义
@variational_estimator
class BayesianEncoder(nn.Module):
    """
    贝叶斯编码器: 将输入特征映射为隐空间中的概率分布（均值和对数方差）。使用了两层贝叶斯线性网络。
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            BayesianLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc_mu = BayesianLinear(hidden_dim, latent_dim)
        self.fc_log_var = BayesianLinear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = self.net(x)
        mu_z = self.fc_mu(hidden)
        log_var_z = self.fc_log_var(hidden)
        return mu_z, log_var_z


@variational_estimator
class BayesianOrdinalDecoder(nn.Module):
    """
    简化的贝叶斯有序解码器:
    - beta系数是全局共享的、可学习的参数，不再依赖于z。
    - 只有阈值tau是z的函数。
    """
    def __init__(self, latent_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        # beta现在是一个固定的、可学习的参数，而不是一个网络
        self.beta = nn.Parameter(torch.randn(1, latent_dim))

        # 阈值网络保持不变，仍然是z的函数
        self.threshold_net = nn.Sequential(
            BayesianLinear(latent_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            BayesianLinear(hidden_dim, num_classes - 1)
        )

    def forward(self, z):
        beta = self.beta.expand(z.size(0), -1).view(-1, z.size(1), 1)
        ordered_thresholds = self.get_thresholds(z)
        return beta, ordered_thresholds

    def get_thresholds(self, z):
        """将阈值计算封装成一个独立方法"""
        threshold_params = self.threshold_net(z)
        cut1 = threshold_params[:, 0:1]
        log_gaps = threshold_params[:, 1:]
        ordered_thresholds = torch.cumsum(torch.cat([cut1, torch.exp(log_gaps)], dim=1), dim=1)
        return ordered_thresholds


@variational_estimator
class MT_OBVAE(nn.Module):
    """
    完整的多任务有序贝叶斯变分自编码器模型。
    整合了编码器和三个独立的解码器。
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes, dropout):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = BayesianEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder_ttc = BayesianOrdinalDecoder(latent_dim, hidden_dim // 2, num_classes, dropout)
        self.decoder_drac = BayesianOrdinalDecoder(latent_dim, hidden_dim // 2, num_classes, dropout)
        self.decoder_psd = BayesianOrdinalDecoder(latent_dim, hidden_dim // 2, num_classes, dropout)

    def reparameterize(self, mu, log_var):
        """重参数化技巧，用于从隐分布中采样，同时保持梯度可传导"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_z, log_var_z = self.encoder(x)
        z = self.reparameterize(mu_z, log_var_z)

        beta_ttc, tau_ttc = self.decoder_ttc(z)
        beta_drac, tau_drac = self.decoder_drac(z)
        beta_psd, tau_psd = self.decoder_psd(z)

        return {
            "mu_z": mu_z, "log_var_z": log_var_z, "z_sample": z,
            "params_ttc": (beta_ttc, tau_ttc),
            "params_drac": (beta_drac, tau_drac),
            "params_psd": (beta_psd, tau_psd)
        }


# 3. 损失函数与训练流程
def ordinal_probit_nll_loss(z, beta, thresholds, y_true):
    """计算单个任务的有序概率单位模型负对数似然损失"""
    # z: [N, latent_dim], beta: [N, latent_dim, 1], y_true: [N]
    latent_pred = torch.bmm(z.unsqueeze(1), beta).squeeze(-1)  # -> [N, 1]
    latent_pred = latent_pred.squeeze(-1)  # -> [N]

    # 填充无穷边界以便统一处理
    # thresholds 初始形状: [N, 3] -> 填充后: [N, 5]
    padded_thresholds = F.pad(thresholds, (1, 1), mode='constant', value=torch.inf)
    padded_thresholds[:, 0] = -torch.inf

    # 根据真实标签y_true (值为0,1,2,3) 获取对应的上下阈值
    upper_bound_idx = y_true.unsqueeze(1) + 1
    lower_bound_idx = y_true.unsqueeze(1)
    upper_thresholds = torch.gather(padded_thresholds, 1, upper_bound_idx).squeeze(-1)
    lower_thresholds = torch.gather(padded_thresholds, 1, lower_bound_idx).squeeze(-1)

    # 计算 P(y=j) = CDF(tau_{j+1} - z'beta) - CDF(tau_j - z'beta)
    normal_dist = torch.distributions.Normal(0, 1)  # 标准正态分布
    prob_j = normal_dist.cdf(upper_thresholds - latent_pred) - normal_dist.cdf(lower_thresholds - latent_pred)

    # 计算负对数似然
    log_prob = torch.log(prob_j + 1e-9)  # 加epsilon防止log(0)
    return -torch.mean(log_prob)


def train_mt_obvae_model(X_train, y_train, output_dir,
                         num_epochs=100, batch_size=64, lr=1e-3,
                         hidden_dim=64, latent_dim=8, dropout=0.3,
                         kl_latent_final_weight=0.01, anneal_portion=0.5):
    """
    训练MT-OBVAE模型的主函数。
    【核心优化】: 不在总损失中显式添加权重KL项，只对隐空间KL损失进行退火。
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    X_t, y_t = torch.tensor(X_train), torch.tensor(y_train)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MT_OBVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_classes=4, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 计算退火结束的epoch数
    anneal_epochs = int(num_epochs * anneal_portion)

    monitor = TrainingMonitor(output_dir)

    print(f"开始训练 MT-OBVAE 模型... 设备: {device}")
    for epoch in range(num_epochs):
        model.train()
        total_recon_epoch, total_kl_lat_epoch = 0, 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        # 实现隐空间KL退火 (KL Annealing)
        if anneal_epochs > 0:
            kl_latent_current_weight = min(kl_latent_final_weight, (epoch + 1) / anneal_epochs * kl_latent_final_weight)
        else:
            kl_latent_current_weight = kl_latent_final_weight

        for Xb, yb in progress_bar:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()

            outputs = model(Xb)
            z, mu_z, log_var_z = outputs['z_sample'], outputs['mu_z'], outputs['log_var_z']

            # 计算重构损失
            recon_loss = (ordinal_probit_nll_loss(z, outputs['params_ttc'][0], outputs['params_ttc'][1], yb[:, 0]) +
                          ordinal_probit_nll_loss(z, outputs['params_drac'][0], outputs['params_drac'][1], yb[:, 1]) +
                          ordinal_probit_nll_loss(z, outputs['params_psd'][0], outputs['params_psd'][1], yb[:, 2]))

            # 计算隐空间KL散度
            kl_latent_loss = -0.5 * torch.sum(1 + log_var_z - mu_z.pow(2) - log_var_z.exp(), dim=1).mean()

            # 组合总损失 (不再包含权重KL项，blitz库会隐式处理)
            total_loss_batch = recon_loss + kl_latent_current_weight * kl_latent_loss

            total_loss_batch.backward()
            optimizer.step()

            total_recon_epoch += recon_loss.item()
            total_kl_lat_epoch += kl_latent_loss.item()

            progress_bar.set_postfix({
                'Recon Loss': f'{recon_loss.item():.4f}',
                'KL Latent': f'{kl_latent_loss.item():.4f}'
            })

        recon_mean = total_recon_epoch / len(dataloader)
        kl_mean = total_kl_lat_epoch / len(dataloader)
        monitor.log_epoch(epoch + 1, recon_mean, kl_mean, kl_latent_current_weight)

        # 打印每个epoch的平均损失
        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"重构损失: {recon_mean:.4f} | "
              f"隐空间KL: {kl_mean:.4f} | "
              f"隐空间KL权重: {kl_latent_current_weight:.4f}")


    monitor.export()

    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_dir, "mt_obvae.pth"))
    print(f"\nMT-OBVAE 模型已保存至 {os.path.join(output_dir, 'mt_obvae.pth')}")
    return model


# 4. 评估与分析接口
def evaluate_model_mt_obvae(model, X_test, y_test, num_samples=100):
    """评估MT-OBVAE模型性能，使用蒙特卡洛采样来平均预测"""
    task_names_lower = ['ttc', 'drac', 'psd']
    task_names_upper = ['TTC', 'DRAC', 'PSD']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_probas_samples = {task: [] for task in task_names_lower}
    X_t = torch.tensor(X_test).to(device)

    print(f"正在使用 {num_samples} 次蒙特卡洛采样进行评估...")
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="评估采样"):
            outputs = model(X_t)
            z_sample = outputs['z_sample']
            for task in task_names_lower:
                beta, thresholds = outputs[f'params_{task}']
                latent_pred = torch.bmm(z_sample.unsqueeze(1), beta).squeeze(-1).squeeze(-1)

                padded_thresholds = F.pad(thresholds, (1, 1), 'constant', torch.inf)
                padded_thresholds[:, 0] = -torch.inf

                cdf_vals = torch.distributions.Normal(0, 1).cdf(padded_thresholds - latent_pred.unsqueeze(1))
                probs = cdf_vals[:, 1:] - cdf_vals[:, :-1]
                all_probas_samples[task].append(probs.cpu().numpy())

    final_probas = []
    for task in task_names_lower:
        task_probas_np = np.stack(all_probas_samples[task], axis=0)
        final_probas.append(np.mean(task_probas_np, axis=0))

    final_probas_stacked = np.stack(final_probas, axis=1)
    metrics = evaluate_multitask_predictions(y_true=y_test, probas=final_probas_stacked, task_names=task_names_upper)
    print("\n评估结果:")
    print(format_results_table(metrics))
    return metrics


def extract_latent_space(model, X_data, y_data, feature_names, batch_size=256):
    """接口 1 & 2: 提取隐空间表示 (用于冲突相空间) 和 风险模糊度 (偶然不确定性)"""
    print("接口 1 & 2: 正在提取隐空间数据...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    results = []

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_data), torch.tensor(y_data))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for i, (Xb, yb) in enumerate(tqdm(dataloader, desc="提取隐空间")):
            Xb = Xb.to(device)
            mu_z, log_var_z = model.encoder(Xb)

            mu_z_cpu, log_var_z_cpu = mu_z.cpu().numpy(), log_var_z.cpu().numpy()

            for j in range(len(Xb)):
                sample_data = {"sample_id": i * batch_size + j}
                sample_data.update({f"mu_z_{k}": mu_z_cpu[j, k] for k in range(model.latent_dim)})
                sample_data.update({f"log_var_z_{k}": log_var_z_cpu[j, k] for k in range(model.latent_dim)})
                sample_data.update({f"{name}": Xb[j, k].cpu().item() for k, name in enumerate(feature_names)})
                sample_data.update({"y_ttc": yb[j, 0].item(), "y_drac": yb[j, 1].item(), "y_psd": yb[j, 2].item()})
                results.append(sample_data)

    df = pd.DataFrame(results)
    # 计算风险模糊度指数 (偶然不确定性的量化)
    var_cols = [f'log_var_z_{k}' for k in range(model.latent_dim)]
    df['risk_ambiguity_index'] = np.sum(np.exp(df[var_cols].values), axis=1)

    return df


def predict_with_uncertainty(model, x_sample, num_mc_samples=100):
    """接口 3: 对单个样本进行预测，并分解不确定性"""
    print("接口 3: 正在对单个样本进行不确定性分解...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    task_names = ['ttc', 'drac', 'psd']
    x_tensor = torch.tensor(x_sample).unsqueeze(0).to(device)

    # 1. 获取偶然不确定性指标 (风险模糊度)
    with torch.no_grad():
        mu_z, log_var_z = model.encoder(x_tensor)
        aleatoric_index = torch.sum(torch.exp(log_var_z)).item()

    # 2. 通过蒙特卡洛采样获取认知不确定性
    all_prob_preds = {task: [] for task in task_names}
    with torch.no_grad():
        for _ in range(num_mc_samples):
            outputs = model(x_tensor)
            z_sample = outputs['z_sample']
            for task in task_names:
                beta, thresholds = outputs[f'params_{task}']
                latent_pred = torch.bmm(z_sample.unsqueeze(1), beta).squeeze(-1).squeeze(-1)
                padded_thresholds = F.pad(thresholds, (1, 1), 'constant', torch.inf)
                padded_thresholds[:, 0] = -torch.inf
                cdf_vals = torch.distributions.Normal(0, 1).cdf(padded_thresholds - latent_pred.unsqueeze(1))
                probs = cdf_vals[:, 1:] - cdf_vals[:, :-1]
                all_prob_preds[task].append(probs.cpu().numpy())

    # 3. 汇总结果
    predictive_mean = {}
    epistemic_uncertainty = {}
    for task in task_names:
        preds_array = np.concatenate(all_prob_preds[task], axis=0)
        predictive_mean[task] = preds_array.mean(axis=0)

        # 认知不确定性可以用多种方式量化, 这里用预测概率向量的方差之和作为示例
        epistemic_uncertainty[task] = np.var(preds_array, axis=0).sum()

    return {
        "平均预测概率": predictive_mean,
        "认知不确定性(方差和)": epistemic_uncertainty,
        "偶然不确定性指数(风险模糊度)": aleatoric_index
    }


def extract_decoder_sensitivities(model):
    """接口 4: 提取解码器敏感性（beta系数网络的权重均值）"""
    print("接口 4: 正在提取解码器敏感性剖面...")
    sensitivities = {}
    task_names = ['ttc', 'drac', 'psd']

    for task in task_names:
        decoder = getattr(model, f"decoder_{task}")
        # 提取beta_layer层的权重均值
        beta_mean_weights = decoder.beta.data.detach().cpu().numpy()
        sensitivities[task] = beta_mean_weights

    return sensitivities


# 5. 主函数
def main():
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    # 文件路径参数
    ap.add_argument("--train", default=f"../data/{ratio_name}/train.csv", help="训练数据路径")
    ap.add_argument("--test", default=f"../data/{ratio_name}/test.csv", help="测试数据路径")
    ap.add_argument("--out_dir", default=f"../output/{ratio_name}/results_mt_obvae", help="输出目录")
    # 训练超参数
    ap.add_argument("--epochs", type=int, default=200, help="训练轮次")
    ap.add_argument("--batch_size", type=int, default=128, help="批处理大小")
    ap.add_argument("--lr", type=float, default=1e-3, help="学习率")
    # 模型结构超参数
    ap.add_argument("--hidden_dim", type=int, default=64, help="隐藏层维度")
    ap.add_argument("--latent_dim", type=int, default=8, help="隐空间维度")
    ap.add_argument("--dropout", type=float, default=0.05, help="Dropout比率")
    # VAE损失超参数
    ap.add_argument("--kl_weight", type=float, default=0.01, help="隐空间KL损失的最终权重")
    ap.add_argument("--anneal_portion", type=float, default=0.5, help="KL退火周期占比")
    ap.add_argument("--density_feature", default=None, help="用于相空间分色的密度特征名")
    ap.add_argument("--uncertainty_indices", default="0", help="逗号分隔的样本索引用于不确定性可视化")
    ap.add_argument("--uncertainty_mc_samples", type=int, default=100, help="不确定性可视化时的MC采样次数")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train, X_test, y_test, feature_names = load_data(args.train, args.test)

    # 训练模型
    model = train_mt_obvae_model(X_train, y_train, args.out_dir,
                                 num_epochs=args.epochs,
                                 batch_size=args.batch_size,
                                 lr=args.lr,
                                 hidden_dim=args.hidden_dim,
                                 latent_dim=args.latent_dim,
                                 dropout=args.dropout,
                                 kl_latent_final_weight=args.kl_weight,
                                 anneal_portion=args.anneal_portion)

    # 评估模型
    metrics = evaluate_model_mt_obvae(model, X_test, y_test, num_samples=50)

    # 保存评估结果
    results_file = os.path.join(args.out_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        # 手动写入表头
        f.write("Task,Accuracy,F1,QWK,OrdMAE,NLL,Brier,AUROC,BrdECE\n")
        # 调用函数获取结果字符串，不再传递header参数
        f.write(format_results_table(metrics))
    print(f"评估结果已保存至 {results_file}")

    # --- 运行并保存所有交通机理分析接口的结果 ---
    print("\n" + "=" * 30)
    print("--- 开始运行并保存交通机理分析接口 ---")
    print("=" * 30 + "\n")

    # 接口 1 & 2: 提取隐空间数据和风险模糊度
    latent_space_df = extract_latent_space(model, X_test, y_test, feature_names)
    latent_space_df_path = os.path.join(args.out_dir, "latent_space_data.csv")
    latent_space_df.to_csv(latent_space_df_path, index=False)
    print(f"\n接口1&2: 隐空间数据已保存至 {latent_space_df_path}")
    print("隐空间数据预览:")
    print(latent_space_df.head())

    # 接口 3: 对单个样本进行不确定性分解
    index_list = [int(idx.strip()) for idx in args.uncertainty_indices.split(',') if idx.strip().isdigit()]
    index_list = [idx for idx in index_list if 0 <= idx < len(X_test)]
    if index_list:
        uncertainty_results = visualize_uncertainty_cases(
            model,
            X_test,
            index_list,
            args.out_dir,
            num_mc_samples=args.uncertainty_mc_samples,
        )
        uncertainty_json = os.path.join(args.out_dir, "uncertainty_cases.json")
        with open(uncertainty_json, "w", encoding="utf-8") as f:
            json.dump(uncertainty_results, f, indent=2, ensure_ascii=False)

        flattened_rows = []
        for item in uncertainty_results:
            for task, stats in item["summaries"].items():
                row = {
                    "sample_index": item["sample_index"],
                    "aleatoric_index": item.get("aleatoric_index"),
                    "task": task,
                    "epistemic_variance": stats["epistemic_variance"],
                    "mutual_information": stats["mutual_information"],
                }
                for i, prob in enumerate(stats["predictive_mean"]):
                    row[f"predictive_mean_class{i}"] = prob
                flattened_rows.append(row)

        pd.DataFrame(flattened_rows).to_csv(os.path.join(args.out_dir, "uncertainty_cases.csv"), index=False)
        print(f"\n接口3: 不确定性案例分析结果已保存至 {uncertainty_json}")
    else:
        print("未提供有效的样本索引，跳过不确定性可视化。")

    # 接口 4: 提取解码器敏感性
    decoder_weights = extract_decoder_sensitivities(model)
    sensitivities_file = os.path.join(args.out_dir, "decoder_sensitivities.npz")
    # 使用np.savez保存多个数组到一个文件
    np.savez(sensitivities_file, **decoder_weights)
    visualize_decoder_sensitivities(sensitivities_file, args.out_dir)
    print(f"\n接口4: 解码器敏感性权重已保存至 {sensitivities_file}")
    for task, weights in decoder_weights.items():
        print(f"--- 任务: {task}, 权重形状: {weights.shape} ---")

    # 生成冲突相空间可视化与风险模糊度分析
    generate_phase_space(latent_space_df_path, args.out_dir, density_feature=args.density_feature)


if __name__ == "__main__":
    main()
