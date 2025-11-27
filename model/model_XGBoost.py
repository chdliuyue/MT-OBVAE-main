import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
from sklearn.model_selection import GridSearchCV

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
    # 读取训练数据和测试数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 在转换为NumPy数组之前，先获取特征名称
    # feature_names = train_data.columns[:-3].tolist() # 在这里选特征
    feature_names = ['UF', 'UAS', 'UD', 'UAL', 'DF', 'DAS', 'DD', 'DAL', 'rq_rel', 'rk_rel', 'CV_v', 'E_BRK']

    # 提取特征和标签，并转换为NumPy数组
    # X_train = train_data.iloc[:, :-3].values
    # X_test = test_data.iloc[:, :-3].values
    X_train = train_data[feature_names].values
    X_test = test_data[feature_names].values

    y_train = train_data.iloc[:, -3:].values
    y_test = test_data.iloc[:, -3:].values

    return X_train, y_train, X_test, y_test, feature_names


# 构建并训练XGBoost模型
def train_xgboost_model(X, y, output_dir, feature_names, random_state: int = 42):
    """
    为每个任务使用网格搜索训练一个XGBoost分类器，并保存特征重要性表。
    """
    models = []
    feature_importances_matrix = []
    task_names = ['TTC', 'DRAC', 'PSD']  # 显示任务对应的标签名

    # 定义要搜索的超参数网格
    # 这是一个示例网格，您可以根据需要调整范围
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.075, 0.1],
        'n_estimators': [100, 150, 200],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0]  # L2 正则化
    }

    for task_idx, task_name in enumerate(task_names):
        print(f"--- Starting GridSearchCV for task: {task_name} ---")

        # 准备每个任务的数据
        y_task = y[:, task_idx]

        # 初始化基础模型 (移除了 use_label_encoder)
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=random_state
        )

        # 设置GridSearchCV
        # cv=3 表示3折交叉验证，n_jobs=-1 使用所有CPU核心
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        # 运行网格搜索
        grid_search.fit(X, y_task)

        print(f"Best parameters found for task {task_name}: {grid_search.best_params_}")

        # 获取最佳模型
        best_model = grid_search.best_estimator_
        models.append(best_model)

        # 获取并存储特征重要性
        importances = best_model.feature_importances_
        feature_importances_matrix.append(importances)

    # 将所有任务的特征重要性保存为一个 CSV 文件
    feature_importances_matrix = np.array(feature_importances_matrix).T

    feature_importances_df = pd.DataFrame(
        feature_importances_matrix,
        columns=task_names,
        index=feature_names
    )

    # 保存特征重要性文件
    output_path = os.path.join(output_dir, 'xgb_importances.csv')
    feature_importances_df.to_csv(output_path, index_label='feature')
    print(f"Level 0 feature importances saved to {output_path}")

    return models


# 评估模型
def evaluate_model(models, X_test, y_test):
    """
    评估每个任务的XGBoost模型性能。
    """
    task_names = ['TTC', 'DRAC', 'PSD']
    all_probas = []

    for i, model in enumerate(models):
        task_name = task_names[i]
        print(f"Evaluating model for task {task_name}...")

        # 使用 predict_proba 获取预测概率
        probas = model.predict_proba(X_test)

        if probas.ndim != 2:
            raise ValueError(f"Expected 2-D probability array, got shape {probas.shape}")

        all_probas.append(probas)

    # 将所有任务的概率堆叠成一个 [N, M, C] 的数组
    all_probas = np.stack(all_probas, axis=1)

    # 使用导入的评估函数计算指标
    metrics = evaluate_multitask_predictions(y_true=y_test, probas=all_probas, task_names=task_names)

    print(format_results_table(metrics))

    return metrics


# 主函数
def main():
    # 设置输入路径、输出路径等参数
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=f"../data/{ratio_name}/train_old.csv")
    ap.add_argument("--test", default=f"../data/{ratio_name}/test_old.csv")
    ap.add_argument("--out_dir", default=f"../output/{ratio_name}/results_xgb_gs")
    ap.add_argument("--runs", type=int, default=5, help="重复实验次数")
    ap.add_argument(
        "--base_seed", type=int, default=42, help="随机种子（每次运行递增）"
    )
    args = ap.parse_args()

    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)

    # 载入数据，并获取特征名称
    X_train, y_train, X_test, y_test, feature_names = load_data(args.train, args.test)

    all_runs = []
    for run_idx in range(args.runs):
        run_dir = os.path.join(args.out_dir, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        current_seed = args.base_seed + run_idx
        set_global_seed(current_seed)

        # 训练XGBoost模型，并保存特征重要性
        models = train_xgboost_model(
            X_train, y_train, run_dir, feature_names, random_state=current_seed
        )

        # 评估模型性能
        metrics = evaluate_model(models, X_test, y_test)
        all_runs.append(metrics)

        # 保存评估结果
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

