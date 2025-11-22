import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 确保可以从父目录导入模块
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from module.metrics import evaluate_multitask_predictions, format_results_table


# 读取和准备数据
def load_data(train_path, test_path):
    """
    读取训练和测试数据，并返回NumPy数组以及特征名称。
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 特征名称
    feature_names = ['UF', 'UAS', 'UD', 'UAL', 'DF', 'DAS', 'DD', 'DAL',
                     'rq_rel', 'rk_rel', 'CV_v', 'E_BRK']

    # 提取特征和标签
    # X_train = train_data.iloc[:, :-3].values
    # X_test = test_data.iloc[:, :-3].values
    X_train = train_data[feature_names].values
    X_test = test_data[feature_names].values

    y_train = train_data.iloc[:, -3:].values
    y_test = test_data.iloc[:, -3:].values

    return X_train, y_train, X_test, y_test, feature_names


# 构建并训练SVM模型
def train_svm_model(X, y, output_dir):
    """
    为每个任务使用网格搜索训练一个SVM分类器。
    """
    models = []
    task_names = ['TTC', 'DRAC', 'PSD']

    # 定义超参数搜索空间
    param_grid = {
        'C': [1, 5],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
    }

    for task_idx, task_name in enumerate(task_names):
        print(f"--- Starting GridSearchCV for task: {task_name} ---")
        y_task = y[:, task_idx]

        # 初始化SVM模型
        model = SVC(probability=True, random_state=42)

        # 设置网格搜索
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        grid_search.fit(X, y_task)
        print(f"Best parameters found for task {task_name}: {grid_search.best_params_}")

        best_model = grid_search.best_estimator_
        models.append(best_model)

    print("All SVM models trained successfully.")
    return models


# 评估模型
def evaluate_model(models, X_test, y_test):
    """
    评估每个任务的SVM模型性能。
    """
    task_names = ['TTC', 'DRAC', 'PSD']
    all_probas = []

    for i, model in enumerate(models):
        task_name = task_names[i]
        print(f"Evaluating model for task {task_name}...")

        probas = model.predict_proba(X_test)
        if probas.ndim != 2:
            raise ValueError(f"Expected 2-D probability array, got shape {probas.shape}")

        all_probas.append(probas)

    all_probas = np.stack(all_probas, axis=1)
    metrics = evaluate_multitask_predictions(y_true=y_test, probas=all_probas, task_names=task_names)

    print(format_results_table(metrics))
    return metrics


# 主函数
def main():
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=f"../data/{ratio_name}/train_old.csv")
    ap.add_argument("--test", default=f"../data/{ratio_name}/test_old.csv")
    ap.add_argument("--out_dir", default=f"../output/{ratio_name}/results_svm_gs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train, X_test, y_test, feature_names = load_data(args.train, args.test)
    models = train_svm_model(X_train, y_train, args.out_dir)
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
