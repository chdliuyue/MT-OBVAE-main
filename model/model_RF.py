import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 确保可以从父目录导入模块
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


# 构建并训练Random Forest模型
def train_random_forest_model(X, y, output_dir, feature_names, random_state: int = 42):
    """
    为每个任务使用网格搜索训练一个Random Forest分类器，并保存特征重要性表。
    """
    models = []
    feature_importances_matrix = []
    task_names = ['TTC', 'DRAC', 'PSD']

    # 定义超参数搜索空间
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 6, 7, 8],
        'min_samples_split': [4],
        'min_samples_leaf': [2],
        'max_features': ['sqrt']
    }

    for task_idx, task_name in enumerate(task_names):
        print(f"--- Starting GridSearchCV for task: {task_name} ---")
        y_task = y[:, task_idx]

        # 初始化基础模型
        model = RandomForestClassifier(random_state=random_state, class_weight='balanced')

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

        # 获取特征重要性
        importances = best_model.feature_importances_
        feature_importances_matrix.append(importances)

    # 保存特征重要性
    feature_importances_matrix = np.array(feature_importances_matrix).T
    feature_importances_df = pd.DataFrame(
        feature_importances_matrix,
        columns=task_names,
        index=feature_names
    )

    output_path = os.path.join(output_dir, 'rf_importances.csv')
    feature_importances_df.to_csv(output_path, index_label='feature')
    print(f"Level 0 feature importances saved to {output_path}")

    return models


# 评估模型
def evaluate_model(models, X_test, y_test):
    """
    评估每个任务的Random Forest模型性能。
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
    ap.add_argument("--out_dir", default=f"../output/{ratio_name}/results_rf_gs")
    ap.add_argument("--runs", type=int, default=5, help="重复实验次数")
    ap.add_argument(
        "--base_seed", type=int, default=42, help="随机种子（每次运行递增）"
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train, X_test, y_test, feature_names = load_data(args.train, args.test)

    all_runs = []
    for run_idx in range(args.runs):
        run_dir = os.path.join(args.out_dir, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        current_seed = args.base_seed + run_idx
        set_global_seed(current_seed)

        models = train_random_forest_model(
            X_train, y_train, run_dir, feature_names, random_state=current_seed
        )
        metrics = evaluate_model(models, X_test, y_test)
        all_runs.append(metrics)

        results_file = os.path.join(run_dir, "evaluation_results.txt")
        with open(results_file, 'w') as f:
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
