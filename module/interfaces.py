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
