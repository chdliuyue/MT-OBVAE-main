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


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

        self.prior = torch.distributions.Normal(0, prior_sigma)

    def sample_weights(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        weight_eps = torch.randn_like(weight_sigma)
        bias_eps = torch.randn_like(bias_sigma)
        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps
        return weight, bias

    def kl_divergence(self, weight, bias):
        weight_log_q = torch.distributions.Normal(self.weight_mu, torch.log1p(torch.exp(self.weight_rho))).log_prob(weight)
        bias_log_q = torch.distributions.Normal(self.bias_mu, torch.log1p(torch.exp(self.bias_rho))).log_prob(bias)
        weight_log_p = self.prior.log_prob(weight)
        bias_log_p = self.prior.log_prob(bias)
        return (weight_log_q - weight_log_p).sum() + (bias_log_q - bias_log_p).sum()

    def forward(self, x):
        weight, bias = self.sample_weights()
        return torch.nn.functional.linear(x, weight, bias), self.kl_divergence(weight, bias)


class BayesianMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim // 2)
        self.fc3 = BayesianLinear(hidden_dim // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        out1, kl1 = self.fc1(x)
        out1 = self.relu(out1)
        out1 = self.bn1(out1)
        out1 = self.dropout(out1)
        out2, kl2 = self.fc2(out1)
        out2 = self.relu(out2)
        logits, kl3 = self.fc3(out2)
        total_kl = kl1 + kl2 + kl3
        return logits, total_kl


# 3. 训练单任务Bayesian MLP模型
def train_bnn_model(
    X_train, y_train, output_dir, num_epochs=100, batch_size=64, lr=1e-3, kl_scale=1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    task_names = ['TTC', 'DRAC', 'PSD']
    models = []

    for task in range(y_train.shape[1]):
        print(f"Training Bayesian MLP for task {task_names[task]}...")
        y_task = y_train[:, task]

        X_t = torch.tensor(X_train)
        y_t = torch.tensor(y_task)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = BayesianMLPClassifier(
            input_dim=input_dim, hidden_dim=128, num_classes=len(np.unique(y_task))
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for Xb, yb in dataloader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits, kl = model(Xb)
                ce_loss = criterion(logits, yb)
                loss = ce_loss + kl_scale * kl
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {total_loss/len(dataloader):.4f}")

        models.append(model)
        torch.save(model.state_dict(), os.path.join(output_dir, f"bnn_{task_names[task]}.pth"))
        print(f"Model for {task_names[task]} saved.")

    return models


# 4. 评估模型
def evaluate_model(models, X_test, y_test, mc_samples: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_names = ['TTC', 'DRAC', 'PSD']
    all_probas = []

    X_t = torch.tensor(X_test).to(device)

    for task, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            mc_logits = []
            for _ in range(mc_samples):
                logits, _ = model(X_t)
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
    ap.add_argument("--out_dir", default="../output/" + ratio_name + "/results_bnn")
    ap.add_argument("--runs", type=int, default=5, help="重复实验次数")
    ap.add_argument(
        "--base_seed", type=int, default=42, help="随机种子（每次运行递增）",
    )
    ap.add_argument("--kl_scale", type=float, default=1e-4, help="KL散度系数")
    ap.add_argument("--mc_samples", type=int, default=5, help="评估时的蒙特卡洛采样次数")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data(args.train, args.test)

    all_runs = []
    for run_idx in range(args.runs):
        run_dir = os.path.join(args.out_dir, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        current_seed = args.base_seed + run_idx
        set_global_seed(current_seed)

        models = train_bnn_model(
            X_train, y_train, run_dir, kl_scale=args.kl_scale
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
