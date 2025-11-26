import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


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
    var_cols = [f"log_var_z_{k}" for k in range(model.latent_dim)]
    df["risk_ambiguity_index"] = np.sum(np.exp(df[var_cols].values), axis=1)

    return df


def predict_with_uncertainty(model, x_sample, num_mc_samples=100):
    """接口 4: 对单个样本进行预测，并分解不确定性"""
    print("接口 4: 正在对单个样本进行不确定性分解...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    task_names = ["ttc", "drac", "psd"]
    x_tensor = torch.tensor(x_sample).unsqueeze(0).to(device)

    with torch.no_grad():
        mu_z, log_var_z = model.encoder(x_tensor)
        aleatoric_index = torch.sum(torch.exp(log_var_z)).item()

    all_prob_preds = {task: [] for task in task_names}
    with torch.no_grad():
        for _ in range(num_mc_samples):
            outputs = model(x_tensor)
            z_sample = outputs["z_sample"]
            for task in task_names:
                beta, thresholds = outputs[f"params_{task}"]
                latent_pred = torch.bmm(z_sample.unsqueeze(1), beta).squeeze(-1).squeeze(-1)
                padded_thresholds = F.pad(thresholds, (1, 1), "constant", torch.inf)
                padded_thresholds[:, 0] = -torch.inf
                cdf_vals = torch.distributions.Normal(0, 1).cdf(padded_thresholds - latent_pred.unsqueeze(1))
                probs = cdf_vals[:, 1:] - cdf_vals[:, :-1]
                all_prob_preds[task].append(probs.cpu().numpy())

    predictive_mean = {}
    epistemic_uncertainty = {}
    for task in task_names:
        preds_array = np.concatenate(all_prob_preds[task], axis=0)
        predictive_mean[task] = preds_array.mean(axis=0)
        epistemic_uncertainty[task] = np.var(preds_array, axis=0).sum()

    return {
        "平均预测概率": predictive_mean,
        "认知不确定性(方差和)": epistemic_uncertainty,
        "偶然不确定性指数(风险模糊度)": aleatoric_index,
    }


def extract_decoder_sensitivities(model):
    """接口 3: 提取解码器敏感性（beta系数网络的权重均值）"""
    print("接口 3: 正在提取解码器敏性剖面...")
    sensitivities = {}
    task_names = ["ttc", "drac", "psd"]

    for task in task_names:
        decoder = getattr(model, f"decoder_{task}")
        beta_mean_weights = decoder.beta.data.detach().cpu().numpy()
        sensitivities[task] = beta_mean_weights

    return sensitivities
