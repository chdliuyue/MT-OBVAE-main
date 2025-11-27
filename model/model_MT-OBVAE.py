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

from module.interfaces import (
    extract_decoder_sensitivities,
    extract_latent_space,
    predict_with_uncertainty,
)
from module.metrics import evaluate_multitask_predictions, format_results_table
from module.phase_space import generate_phase_space
from module.sensitivity_profile import visualize_decoder_sensitivities
from module.training_monitor import TrainingMonitor
from module.uncertainty_viz import rank_uncertainty_extremes, visualize_uncertainty_cases

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


def metrics_to_dataframe(metrics):
    rows = []
    for m in metrics:
        rows.append({
            "task": m.task,
            "accuracy": m.accuracy,
            "f1_score": m.f1_score,
            "qwk": m.qwk,
            "ordmae": m.ordmae,
            "nll": m.nll,
            "brier": m.brier,
            "auroc": m.auroc,
            "brdece": m.brdece,
        })
    return pd.DataFrame(rows)


def summarize_metric_runs(metric_runs):
    frames = []
    for idx, run in enumerate(metric_runs, start=1):
        df = metrics_to_dataframe(run).copy()
        df.insert(0, "run", idx)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    metric_cols = [col for col in combined.columns if col not in {"task", "run"}]
    summary = combined.groupby("task")[metric_cols].agg(['mean', 'std'])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    return combined, summary


def select_representative_high_risk_samples(y_labels: np.ndarray, top_k: int = 5) -> list[int]:
    """选择三种风险指标均为高风险的代表性样本索引。"""

    severe_mask = (y_labels >= 2).sum(axis=1) == y_labels.shape[1]
    severe_indices = np.where(severe_mask)[0]
    if len(severe_indices) == 0:
        scores = y_labels.sum(axis=1)
        severe_indices = np.argsort(-scores)

    return severe_indices[:top_k].tolist()



# 5. 主函数
def main():
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    # 文件路径参数
    ap.add_argument("--train", default=f"../data/{ratio_name}/train.csv", help="训练数据路径")
    ap.add_argument("--test", default=f"../data/{ratio_name}/test.csv", help="测试数据路径")
    ap.add_argument("--out_dir", default=f"../output/{ratio_name}/results_mt_obvae", help="输出目录")
    # 训练超参数
    ap.add_argument("--epochs", type=int, default=2, help="训练轮次")
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
    ap.add_argument("--embedding_method", default="both", choices=["tsne", "umap", "both"], help="冲突相空间降维方式")
    ap.add_argument("--embedding_perplexity", type=int, default=30, help="t-SNE困惑度/邻域大小")
    ap.add_argument("--uncertainty_indices", default="", help="逗号分隔的样本索引用于不确定性可视化，留空自动选择")
    ap.add_argument("--uncertainty_top_k", type=int, default=5, help="自动选择高风险样本的数量")
    ap.add_argument("--uncertainty_mc_samples", type=int, default=100, help="不确定性可视化时的MC采样次数")
    ap.add_argument("--eval_mc_samples", type=int, default=50, help="评估时的蒙特卡洛采样次数")
    ap.add_argument("--eval_runs", type=int, default=3, help="重复评估次数以统计均值和标准差")
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

    # 评估模型（多次运行以统计均值和标准差）
    metric_runs = []
    for eval_idx in range(args.eval_runs):
        print(f"\n评估轮次 {eval_idx + 1}/{args.eval_runs}")
        metrics = evaluate_model_mt_obvae(model, X_test, y_test, num_samples=args.eval_mc_samples)
        metric_runs.append(metrics)

    per_run_df, summary_df = summarize_metric_runs(metric_runs)
    print("\n多次评估均值与标准差：")
    print(summary_df.to_string(index=False))

    # 保存评估结果
    results_file = os.path.join(args.out_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("Task,Accuracy,F1,QWK,OrdMAE,NLL,Brier,AUROC,BrdECE\n")
        f.write(format_results_table(metric_runs[0]))
        f.write("\n\nMean/Std over runs (csv format stored separately)")
    per_run_df.to_csv(os.path.join(args.out_dir, "evaluation_runs.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out_dir, "evaluation_summary.csv"), index=False)
    print(f"评估结果已保存至 {results_file}")

    # --- 运行并保存所有交通机理分析接口的结果 ---
    print("\n" + "=" * 30)
    print("--- 开始运行并保存交通机理分析接口 ---")
    print("=" * 30 + "\n")

    # 接口 1 & 2: 提取隐空间数据和风险模糊度
    latent_space_df = extract_latent_space(model, X_test, y_test, feature_names)
    latent_space_df_path = os.path.join(args.out_dir, "1_latent_space_data.csv")
    latent_space_df.to_csv(latent_space_df_path, index=False)
    print(f"\n接口1&2: 隐空间数据已保存至 {latent_space_df_path}")
    print("隐空间数据预览:")
    print(latent_space_df.head())

    # 接口 1&2：生成冲突相空间可视化与风险模糊度分析
    generate_phase_space(
        latent_space_df_path,
        args.out_dir,
        density_feature=args.density_feature,
        prefix="2_",
        embedding_method=args.embedding_method,
        perplexity=args.embedding_perplexity,
        latent_prefix="1_",
    )

    # 接口 3: 风险敏感性剖面与触发机制解耦
    decoder_weights = extract_decoder_sensitivities(model)
    sensitivities_file = os.path.join(args.out_dir, "3_decoder_sensitivities.npz")
    np.savez(sensitivities_file, **decoder_weights)
    visualize_decoder_sensitivities(
        sensitivities_file,
        args.out_dir,
        prefix="3_",
        sort_dimensions=False,
        latent_df=latent_space_df,
    )
    print(f"\n接口3: 解码器敏感性权重已保存至 {sensitivities_file}")
    for task, weights in decoder_weights.items():
        print(f"--- 任务: {task}, 权重形状: {weights.shape} ---")

    # 接口 4: 认知不确定性与可靠性感知预警
    index_list = [int(idx.strip()) for idx in args.uncertainty_indices.split(',') if idx.strip().isdigit()]
    index_list = [idx for idx in index_list if 0 <= idx < len(X_test)]
    uncertainty_metric = "mutual_information"
    if not index_list:
        high_uncertainty, low_uncertainty, ranking_df = rank_uncertainty_extremes(
            model,
            X_test,
            num_mc_samples=args.uncertainty_mc_samples,
            top_k=args.uncertainty_top_k,
            metric=uncertainty_metric,
            output_dir=args.out_dir,
        )
        index_list = high_uncertainty + [idx for idx in low_uncertainty if idx not in high_uncertainty]

    if index_list:
        uncertainty_results = visualize_uncertainty_cases(
            model,
            X_test,
            index_list,
            args.out_dir,
            num_mc_samples=args.uncertainty_mc_samples,
        )
        uncertainty_json = os.path.join(args.out_dir, "4_uncertainty_cases.json")
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

        pd.DataFrame(flattened_rows).to_csv(os.path.join(args.out_dir, "4_uncertainty_cases.csv"), index=False)
        print(f"\n接口4: 不确定性案例分析结果已保存至 {uncertainty_json}")
    else:
        print("未找到满足条件的高风险样本，跳过不确定性可视化。")


if __name__ == "__main__":
    main()
