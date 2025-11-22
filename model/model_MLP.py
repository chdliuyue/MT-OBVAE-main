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

from module.metrics import evaluate_multitask_predictions, format_results_table


# 1. 数据加载
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.iloc[:, :-3].values.astype(np.float32)
    y_train = train_data.iloc[:, -3:].values.astype(np.int64)
    X_test = test_data.iloc[:, :-3].values.astype(np.float32)
    y_test = test_data.iloc[:, -3:].values.astype(np.int64)

    return X_train, y_train, X_test, y_test


# 2. 定义MLP模型
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# 3. 训练单任务MLP模型
def train_mlp_model(X_train, y_train, output_dir, num_epochs=100, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    task_names = ['TTC', 'DRAC', 'PSD']
    models = []

    for task in range(y_train.shape[1]):
        print(f"Training MLP for task {task_names[task]}...")
        y_task = y_train[:, task]

        # 转为Tensor
        X_t = torch.tensor(X_train)
        y_t = torch.tensor(y_task)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 定义模型
        model = MLPClassifier(input_dim=input_dim, hidden_dim=128, num_classes=len(np.unique(y_task))).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练循环
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for Xb, yb in dataloader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {total_loss/len(dataloader):.4f}")

        models.append(model)
        torch.save(model.state_dict(), os.path.join(output_dir, f"mlp_{task_names[task]}.pth"))
        print(f"Model for {task_names[task]} saved.")

    return models


# 4. 评估模型
def evaluate_model(models, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_names = ['TTC', 'DRAC', 'PSD']
    all_probas = []

    X_t = torch.tensor(X_test).to(device)

    for task, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            logits = model(X_t)
            probas = torch.softmax(logits, dim=1).cpu().numpy()
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
    ap.add_argument("--out_dir", default="../output/" + ratio_name + "/results_mlp")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data(args.train, args.test)

    models = train_mlp_model(X_train, y_train, args.out_dir)

    metrics = evaluate_model(models, X_test, y_test)

    results_file = os.path.join(args.out_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("Task | Accuracy | F1 | QWK | OrdMAE | NLL | Brier | AUROC | BrdECE\n")
        for metric in metrics:
            f.write(
                f"{metric.task} | {metric.accuracy:.4f} | {metric.f1_score:.4f} | "
                f"{metric.qwk:.4f} | {metric.ordmae:.4f} | {metric.nll:.4f} | "
                f"{metric.brier:.4f} | {metric.auroc:.4f} | {metric.brdece:.4f}\n"
            )

    print(f"Evaluation results saved to {results_file}")


if __name__ == "__main__":
    main()
