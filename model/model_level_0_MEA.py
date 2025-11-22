import os
import argparse
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# 由于 evaluate_multitask_predictions 和 format_results_table 不再使用，
# 我们可以注释掉或删除这个导入（除非您将来想重新启用它）
# from module.metrics import evaluate_multitask_predictions, format_results_table


# 读取和准备数据
def load_data(train_path, test_path):
    # 读取训练数据和测试数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    feature_names = ['UF', 'UAS', 'UD', 'UAL', 'DF', 'DAS', 'DD', 'DAL', 'rq_rel', 'rk_rel', 'CV_v', 'E_BRK']

    # 提取特征和标签
    X_train = train_data[feature_names].values
    X_test = test_data[feature_names].values

    y_train = train_data.iloc[:, -3:].values  # 标签（后三列）
    y_test = test_data.iloc[:, -3:].values  # 标签（后三列）

    return X_train, y_train, X_test, y_test, feature_names


# 构建并训练PPO有序Logit模型
def train_ppo_model(X, y, feature_names, output_dir, significance_threshold=0.10):
    """
    使用Partial Proportional Odds（PPO）方法训练有序Logit模型，并保存系数表
    """

    models = []
    coef_matrix = []
    task_names = ['TTC', 'DRAC', 'PSD']  # 显示任务对应的标签名

    for task in range(y.shape[1]):  # 针对每个任务（TTC, DRAC, PSD）训练模型
        print(f"Training model for task {task_names[task]}...")

        # 为PPO方法准备数据
        y_task = y[:, task]

        # 检查类别数量
        unique_classes = np.unique(y_task)
        print(f"Task {task_names[task]} has {len(unique_classes)} classes: {unique_classes}")

        # 训练OrderedModel模型
        # 修正：移除了 exog_names=feature_names 参数，因为它导致了 AttributeError
        model = OrderedModel(y_task, X, distr='probit')
        result = model.fit(method='bfgs')
        models.append(result)

        # 修正：获取所有参数名称（包括特征和阈值/截距）
        # all_param_names = [row[0] for row in result.summary().tables[0].data[1:]] # <-- 这是导致错误的行

        # --- 新的修正方法 ---
        # result.params 是一个 ndarray，包含特征系数，然后是阈值
        # 1. 获取特征数量
        num_features = len(feature_names)  # 这是我们从 load_data 传入的列表

        # 2. 获取总参数数量
        num_params = len(result.params)

        # 3. 计算阈值数量
        num_thresholds = num_params - num_features
        if num_thresholds < 0:
            # 这在逻辑上不应该发生，但作为安全检查
            raise ValueError(
                f"Fewer parameters ({num_params}) than feature names ({num_features})? This shouldn't happen.")

        # 4. 动态生成阈值的名称
        #    为了简单和稳健，我们使用唯一的占位符
        threshold_names = [f'threshold_{i + 1}' for i in range(num_thresholds)]

        # 5. 合并为完整的参数名称列表
        #    顺序：特征系数 + 阈值系数
        all_param_names = feature_names + threshold_names

        # 6. 健全性检查
        if len(all_param_names) != num_params:
            # 这种情况不应该发生，但如果发生了，说明逻辑有误
            raise ValueError(
                f"Fatal error: Parameter name list length ({len(all_param_names)}) "
                f"does not match result.params length ({num_params})."
            )

        # 现在 all_param_names 长度（例如 15）与 result.params 长度（15）匹配
        params_series = pd.Series(result.params, index=all_param_names)
        pvalues_series = pd.Series(result.pvalues, index=all_param_names)

        # 筛选出特征的系数
        # .loc[feature_names] 现在可以正确工作，因为它从 Series 中选取键
        feature_coef_df = pd.DataFrame({
            'coef': params_series.loc[feature_names],
            'p_value': pvalues_series.loc[feature_names]
        })

        # 将P值不显著的系数设为NaN
        feature_coef_df['coef'] = np.where(feature_coef_df['p_value'] > significance_threshold, np.nan,
                                           feature_coef_df['coef'])
        coef_matrix.append(feature_coef_df['coef'].values)

        print(result.summary())

    # 将所有任务的系数表保存为一个 CSV 文件
    coef_matrix = np.array(coef_matrix).T
    coef_matrix_df = pd.DataFrame(coef_matrix, columns=task_names, index=feature_names)

    coef_matrix_df.to_csv(
        os.path.join(output_dir, 'level_0_coefficients.csv'), index_label='feature'
    )
    print(f"Level 0 coefficients saved to {output_dir}/level_0_coefficients.csv")

    return models


# --- 评估模型功能已按要求删除 ---

# --- 新增功能：边际效应分析 ---
def analyze_marginal_effects(models, X_train, feature_names, task_names, feature_to_analyze, output_dir):
    """
    分析并绘制一个特征变化时，每个类别的预测概率变化。
    其他特征保持在其均值。
    """
    print(f"\nAnalyzing marginal effects for feature: {feature_to_analyze}...")

    try:
        # 获取要分析特征的索引
        feature_index = feature_names.index(feature_to_analyze)
    except ValueError:
        print(f"Error: Feature '{feature_to_analyze}' not found in feature list.")
        print(f"Available features: {feature_names}")
        return

    # 1. 计算所有特征的均值
    X_means = np.mean(X_train, axis=0)

    # 2. 获取被分析特征的取值范围
    f_min = np.min(X_train[:, feature_index])
    print(f_min)
    f_max = np.max(X_train[:, feature_index])
    print(f_max)
    # 创建一个包含100个点的平滑范围
    feature_range = np.linspace(f_min, f_max, 100)

    for i, (model, task_name) in enumerate(zip(models, task_names)):
        print(f"Generating plot for task: {task_name}")

        # 3. 创建一个“合成”数据集
        #    - 形状为 (100, num_features)
        #    - 每一行都等于 X_means
        X_synthetic = np.tile(X_means, (100, 1))

        # 4. 将被分析特征的那一列替换为 feature_range
        X_synthetic[:, feature_index] = feature_range

        # 5. 使用模型进行预测，得到 (100, num_classes) 的概率矩阵
        probas = model.predict(X_synthetic)

        if np.isnan(probas).any():
            print(f"Warning: NaN values encountered in predictions for task {task_name}. Skipping plot.")
            continue

        num_classes = probas.shape[1]
        print(f"Task {task_name} has {num_classes} outcome classes for plotting.")

        # 6. 绘图
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams.update({
            'axes.linewidth': 2,
            'image.cmap': 'vlag'  # Set the default colormap for all plots  # coolwarm
        })

        # --- 新增：定义类别标签 ---
        class_labels = {
            0: "Safe (0)",
            1: "Low (1)",
            2: "Medium (2)",
            3: "High (3)"
        }
        # --- 结束新增 ---

        fig, ax = plt.subplots(figsize=(8, 6))
        for c in range(num_classes):
            # 绘制每个类别的概率曲线
            # --- 修改：使用新的标签 ---
            label = class_labels.get(c, f"Class {c}")  # 如果c>3，则回退为 "Class c"
            plt.plot(feature_range, probas[:, c], label=label, lw=6)  # 稍微加粗线条

        # --- 修改：增加 fontsize ---
        # plt.xlabel(f"Value of {feature_to_analyze}", fontsize=14)
        # plt.ylabel("Predicted Probability", fontsize=14)
        # plt.title(f"Predicted Probabilities for {task_name} as {feature_to_analyze} Varies", fontsize=16)

        # --- 修改：增加 frameon=False (无边框) 和 fontsize ---
        plt.legend( frameon=False, fontsize=22)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1)  # 概率范围在 0 到 1 之间

        # 7. 保存图像
        plot_filename = os.path.join(output_dir, f"marginal_effects_{task_name}_{feature_to_analyze}.png")
        plt.savefig(plot_filename)
        plt.close()  # 关闭图像以释放内存

        print(f"Saved plot: {plot_filename}")


# 主函数
def main():
    # 设置输入路径、输出路径等参数
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=f"../data/{ratio_name}/train.csv")
    ap.add_argument("--test", default=f"../data/{ratio_name}/test.csv")
    ap.add_argument("--train_old", default=f"../data/{ratio_name}/train_old.csv")
    ap.add_argument("--test_old", default=f"../data/{ratio_name}/test_old.csv")
    ap.add_argument("--out_dir", default=f"../output/{ratio_name}/results_level_0_MEA")
    # 新增参数：指定要分析的特征
    ap.add_argument("--feature", default="CV_v", help="Feature to analyze for marginal effects (e.g., UF, CV_v)")
    args = ap.parse_args()

    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)

    # 载入数据，并获取 feature_names
    X_train, y_train, X_test, y_test, feature_names = load_data(args.train_old, args.test_old)

    # 训练PPO模型，并保存系数表
    models = train_ppo_model(X_train, y_train, feature_names, args.out_dir)

    # --- 评估模型性能和保存评估结果的功能已按要求删除 ---

    # --- 新增：执行边际效应分析 ---
    task_names = ['TTC', 'DRAC', 'PSD']  # 确保与 train_ppo_model 中的一致
    analyze_marginal_effects(models, X_train, feature_names, task_names, args.feature, args.out_dir)

    print("\nAnalysis complete. Coefficient CSV and marginal effects plots saved to:")
    print(args.out_dir)


if __name__ == "__main__":
    main()

