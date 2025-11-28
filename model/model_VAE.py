import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from module.metrics import (
    evaluate_multitask_predictions,
    format_results_table,
    format_summary_table,
    summarise_metric_runs,
)
from module.utils import set_global_seed


# 1. 数据加载
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.iloc[:, :-3].values.astype(np.float32)
    y_train = train_data.iloc[:, -3:].values.astype(np.int64)
    X_test = test_data.iloc[:, :-3].values.astype(np.float32)
    y_test = test_data.iloc[:, -3:].values.astype(np.int64)

    return X_train, y_train, X_test, y_test


class VAEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        logits = self.classifier(z)
        return logits, recon_x, mu, logvar


# 3. 训练单任务VAE模型
def train_vae_model(
    X_train,
    y_train,
    output_dir,
    num_epochs=100,
    batch_size=64,
    lr=1e-3,
    latent_dim=16,
    recon_scale=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    task_names = ['TTC', 'DRAC', 'PSD']
    models = []

    for task in range(y_train.shape[1]):
        print(f"Training VAE classifier for task {task_names[task]}...")
        y_task = y_train[:, task]

        X_t = torch.tensor(X_train)
        y_t = torch.tensor(y_task)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = VAEClassifier(
            input_dim=input_dim,
            hidden_dim=128,
            latent_dim=latent_dim,
            num_classes=len(np.unique(y_task)),
        ).to(device)
        criterion_cls = nn.CrossEntropyLoss()
        criterion_recon = nn.MSELoss(reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for Xb, yb in dataloader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()

                logits, recon_x, mu, logvar = model(Xb)
                cls_loss = criterion_cls(logits, yb)
                recon_loss = criterion_recon(recon_x, Xb)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / Xb.size(0)
                loss = cls_loss + recon_scale * recon_loss + 1e-4 * kl_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] | Loss: {total_loss/len(dataloader):.4f}"
                )

        models.append(model)
        torch.save(model.state_dict(), os.path.join(output_dir, f"vae_{task_names[task]}.pth"))
        print(f"Model for {task_names[task]} saved.")

    return models


# 4. 评估模型
def evaluate_model(models, X_test, y_test, mc_samples: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_names = ['TTC', 'DRAC', 'PSD']
    all_probas = []

    X_t = torch.tensor(X_test).to(device)

    for task, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            mc_logits = []
            for _ in range(mc_samples):
                logits, _, _, _ = model(X_t)
                mc_logits.append(torch.softmax(logits, dim=1))
            probas = torch.stack(mc_logits, dim=0).mean(dim=0).cpu().numpy()
        all_probas.append(probas)

    all_probas = np.stack(all_probas, axis=1)  # [N, M, C]
    metrics = evaluate_multitask_predictions(y_true=y_test, probas=all_probas, task_names=task_names)

    print(format_results_table(metrics))
    return metrics


# 5. 主函数
def main():
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/" + ratio_name + "/train_old.csv")
    ap.add_argument("--test", default="../data/" + ratio_name + "/test_old.csv")
    ap.add_argument("--out_dir", default="../output/" + ratio_name + "/results_vae")
    ap.add_argument("--runs", type=int, default=5, help="重复实验次数")
    ap.add_argument(
        "--base_seed", type=int, default=42, help="随机种子（每次运行递增）",
    )
    ap.add_argument("--latent_dim", type=int, default=16, help="潜变量维度")
    ap.add_argument("--recon_scale", type=float, default=1.0, help="重构损失系数")
    ap.add_argument("--mc_samples", type=int, default=1, help="评估时的蒙特卡洛采样次数")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data(args.train, args.test)

    all_runs = []
    for run_idx in range(args.runs):
        run_dir = os.path.join(args.out_dir, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        current_seed = args.base_seed + run_idx
        set_global_seed(current_seed)

        models = train_vae_model(
            X_train,
            y_train,
            run_dir,
            latent_dim=args.latent_dim,
            recon_scale=args.recon_scale,
        )
        metrics = evaluate_model(models, X_test, y_test, mc_samples=args.mc_samples)
        all_runs.append(metrics)

        results_file = os.path.join(run_dir, "evaluation_results.txt")
        with open(results_file, "w") as f:
            f.write("Task | Accuracy | F1 | QWK | OrdMAE | NLL | Brier | AUROC | BrdECE\n")
            for metric in metrics:
                f.write(
                    f"{metric.task} | {metric.accuracy:.4f} | {metric.f1_score:.4f} | "
                    f"{metric.qwk:.4f} | {metric.ordmae:.4f} | {metric.nll:.4f} | "
                    f"{metric.brier:.4f} | {metric.auroc:.4f} | {metric.brdece:.4f}\n"
                )

        print(f"Run {run_idx} results saved to {results_file}")

    _, mean_df, std_df = summarise_metric_runs(all_runs)
    summary_file = os.path.join(args.out_dir, "evaluation_summary.csv")
    mean_df.join(std_df, lsuffix="_mean", rsuffix="_std").to_csv(summary_file)

    print("\nAggregated metrics (mean ± std):")
    print(format_summary_table(mean_df, std_df))
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
