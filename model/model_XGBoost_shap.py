import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

# 确保可以从父目录导入模块
# Make sure modules can be imported from the parent directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# 读取和准备数据
# Read and prepare data
def load_data(train_path, test_path):
    # 读取训练数据和测试数据
    # Read training and testing data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 在转换为NumPy数组之前，先获取特征名称
    # Get feature names before converting to NumPy arrays
    feature_names = train_data.columns[:-3].tolist()  # 在这里选特征 (Select features here)

    # 提取特征和标签，并转换为NumPy数组
    # Extract features and labels, and convert to NumPy arrays
    X_train = train_data.iloc[:, :-3].values
    y_train = train_data.iloc[:, -3:].values
    X_test = test_data.iloc[:, :-3].values
    y_test = test_data.iloc[:, -3:].values

    return X_train, y_train, X_test, y_test, feature_names


# 构建并训练XGBoost模型
def train_xgboost_model(X, y, output_dir, feature_names):
    models = []
    feature_importances_matrix = []
    task_names = ['TTC', 'DRAC', 'PSD']

    # 定义要搜索的超参数网格
    param_grid = {
        'max_depth': [3, 4],
        'learning_rate': [0.05, 0.075, 0.1],
        'n_estimators': [100, 150, 200],
        'reg_lambda': [0.5, 1.5, 2]  # L2 正则化 (L2 regularization)
    }
    # param_grid = {
    #     'max_depth': [3],
    #     'learning_rate': [0.1],
    #     'n_estimators': [100],
    #     'reg_lambda': [0.5]  # L2 正则化 (L2 regularization)
    # }

    for task_idx, task_name in enumerate(task_names):
        print(f"--- Starting GridSearchCV for task: {task_name} ---")

        # 准备每个任务的数据
        # Prepare data for each task
        y_task = y[:, task_idx]

        # 初始化基础模型
        # Initialize the base model
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42
        )

        # 设置GridSearchCV
        # Set up GridSearchCV
        # cv=3 表示3折交叉验证，n_jobs=-1 使用所有CPU核心
        # cv=3 means 3-fold cross-validation, n_jobs=-1 uses all CPU cores
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        # 运行网格搜索
        # Run grid search
        grid_search.fit(X, y_task)

        print(f"Best parameters found for task {task_name}: {grid_search.best_params_}")

        # 获取最佳模型
        # Get the best model
        best_model = grid_search.best_estimator_
        models.append(best_model)

        # 获取并存储特征重要性
        # Get and store feature importances
        importances = best_model.feature_importances_
        feature_importances_matrix.append(importances)

    # 将所有任务的特征重要性保存为一个 CSV 文件
    # Save feature importances for all tasks into a single CSV file
    feature_importances_matrix = np.array(feature_importances_matrix).T

    feature_importances_df = pd.DataFrame(
        feature_importances_matrix,
        columns=task_names,
        index=feature_names
    )

    # 保存特征重要性文件
    # Save the feature importance file
    output_path = os.path.join(output_dir, 'xgb_importances.csv')
    feature_importances_df.to_csv(output_path, index_label='feature')
    print(f"\nLevel 0 feature importances saved to {output_path}")

    return models


def run_final_shap_plots(models, X_test, feature_names, output_dir, top_n_donut=12):
    """
    Generates three specific, professionally styled SHAP plots for each class of each task.
    """
    print("\n--- Starting Final SHAP Plot Generation with Enhanced Aesthetics ---")

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 10,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 30,
        'legend.title_fontsize': 20,
        'axes.linewidth': 2,
        'image.cmap': 'vlag'  # Set the default colormap for all plots  # coolwarm
    })

    task_names = ['TTC', 'DRAC', 'PSD']
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Use Seaborn's professional qualitative palette for the donut chart
    qualitative_palette = sns.color_palette("tab20", len(feature_names))
    feature_color_mapping = {name: color for name, color in zip(feature_names, qualitative_palette)}
    feature_color_mapping['Others'] = '#D3D3D3'

    for i, model in enumerate(models):
        task_name = task_names[i]
        task_output_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)

        print(f"\nProcessing Task: {task_name}...")

        class_labels = model.classes_
        class_names = [f'Class {label}' for label in class_labels]
        print(f"  - Detected model classes: {class_labels}")

        explainer = shap.TreeExplainer(model)
        explanation_all_classes = explainer(X_test_df)

        # --- 步骤一：循环生成所有类别的 Beeswarm 和 Donut 图 ---
        # for c_idx, c_name in enumerate(class_names):
        #     print(f"\n  Generating plots for {c_name} in {task_name}...")
        #     explanation_for_class = explanation_all_classes[:, :, c_idx]
        #
        #     # --- Chart 1: Beeswarm Plot ---
        #     print(f"    - Generating Beeswarm Plot...")
        #     fig, ax = plt.subplots(figsize=(12, 8))
        #     # The 'cmap' argument is removed; it will now inherit the global setting
        #     shap.plots.beeswarm(explanation_for_class, max_display=len(feature_names), show=False)
        #     ax.tick_params(axis='x', labelsize=20)
        #     # plt.title(f'SHAP Beeswarm Plot for {c_name} in {task_name}')
        #     plt.tight_layout()
        #     save_path = os.path.join(task_output_dir, f'shap_beeswarm_{task_name}_{c_name.replace(" ", "")}.png')
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     plt.close()
        #     print(f"      Saved to {save_path}")
        #
        #     # --- Chart 2: Donut Chart & Data Saving ---
        #     print(f"    - Generating Custom Donut Chart...")
        #     mean_abs_shap = np.abs(explanation_for_class.values).mean(axis=0)
        #
        #     # 1. 创建包含所有特征及其重要性的DataFrame
        #     importance_df = pd.DataFrame({
        #         'feature': explanation_for_class.feature_names,
        #         'mean_abs_shap_value': mean_abs_shap
        #     }).sort_values('mean_abs_shap_value', ascending=False)  # 从大到小排序
        #
        #     # 2. 计算百分比并添加到DataFrame中
        #     total_importance = importance_df['mean_abs_shap_value'].sum()
        #     importance_df['percentage'] = (importance_df['mean_abs_shap_value'] / total_importance) * 100
        #
        #     # 3. 定义保存路径并保存为CSV文件
        #     csv_save_path = os.path.join(task_output_dir, f'shap_importance_{task_name}_{c_name.replace(" ", "")}.csv')
        #     importance_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
        #     print(f"      Saved sorted importance data to {csv_save_path}")
        #
        #     # 4. 排序并分组 (逻辑不变)
        #     sorted_indices = np.argsort(mean_abs_shap)[::-1]
        #     sorted_importance = mean_abs_shap[sorted_indices]
        #     sorted_features = np.array(explanation_for_class.feature_names)[sorted_indices]
        #
        #     if len(sorted_importance) > top_n_donut:
        #         top_importance = sorted_importance[:top_n_donut]
        #         other_importance = sorted_importance[top_n_donut:].sum()
        #         final_importance = np.append(top_importance, other_importance)
        #         final_features = np.append(sorted_features[:top_n_donut], "Others")
        #     else:
        #         final_importance = sorted_importance
        #         final_features = sorted_features
        #
        #     non_zero_mask = final_importance > 0
        #     plot_importance = final_importance[non_zero_mask]
        #     plot_features = final_features[non_zero_mask]
        #     plot_colors = [feature_color_mapping.get(f, '#D3D3D3') for f in plot_features]
        #
        #     # 5. 绘图
        #     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))
        #     wedges, _ = ax.pie(
        #         plot_importance, startangle=90,
        #         wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2),
        #         colors=plot_colors, labels=None
        #     )
        #
        #     # 6. 添加中心文本和图例
        #     ax.text(0, 0, 'Importance', ha='center', va='center', fontsize=30, fontweight='bold')
        #     # ax.set_title(f'Feature Importance Donut Chart for {c_name} in {task_name}', pad=20)
        #     ax.legend(wedges, plot_features, loc="center left", bbox_to_anchor=(1.2, 0.5), frameon=False)
        #     fig.tight_layout()
        #     fig.subplots_adjust(right=0.75)  # 确保为图例留出足够空间
        #     save_path = os.path.join(task_output_dir, f'shap_donutchart_{task_name}_{c_name.replace(" ", "")}.png')
        #     fig.savefig(save_path, dpi=300, bbox_inches='tight')
        #     plt.close(fig)
        #     print(f"      Saved to {save_path}")

        # --- 图表3: 特征贡献图 (Dependence Plot) --- (已最终修正)
        print(f"\n  Generating Dependence Plots for all features in {task_name}...")
        for feature in feature_names:
            for c_idx, c_name in enumerate(class_names):
                explanation_for_class = explanation_all_classes[:, :, c_idx]

                # 1. 创建 Figure 和 Axes 对象
                fig, ax = plt.subplots(figsize=(8, 6))

                try:
                    # 2. 将 ax 传递给绘图函数
                    shap.plots.scatter(
                        explanation_for_class[:, feature],
                        color=explanation_for_class,
                        ax=ax,
                        show=False
                    )
                except Exception:
                    shap.plots.scatter(explanation_for_class[:, feature], ax=ax, show=False)

                # *** 在这里应用修改 ***
                # 3. 单独设置主图表的刻度字体大小
                ax.tick_params(axis='x', labelsize=30)
                ax.tick_params(axis='y', labelsize=30)

                # 4. 遍历所有由SHAP创建的坐标轴 (主图、颜色条、直方图)
                #    并为所有非主图的坐标轴设置刻度字体大小
                for sub_ax in fig.axes:
                    if sub_ax != ax:  # 如果不是主图的ax
                        sub_ax.tick_params(labelsize=30)  # 设置它的刻度大小


                # 去掉边框
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                fig.tight_layout()
                save_path = os.path.join(task_output_dir,
                                         f'shap_dependence_{task_name}_{c_name.replace(" ", "")}_{feature}.png')
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

        print(f"    Saved all dependence plots for {task_name}.")

# 主函数
def main():
    # 设置输入路径、输出路径等参数
    # Set parameters like input paths, output directory, etc.
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=f"../data/{ratio_name}/train.csv")
    ap.add_argument("--test", default=f"../data/{ratio_name}/test.csv")
    ap.add_argument("--train_old", default=f"../data/{ratio_name}/train_old.csv")
    ap.add_argument("--test_old", default=f"../data/{ratio_name}/test_old.csv")
    ap.add_argument("--out_dir", default=f"../output/{ratio_name}/results_xgb_shap")
    args = ap.parse_args()

    # 确保输出目录存在
    # Ensure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # 载入数据，并获取特征名称
    # Load data and get feature names
    X_train, y_train, X_test, y_test, feature_names = load_data(args.train_old, args.test_old)

    # 训练XGBoost模型，并保存特征重要性
    # Train XGBoost models and save feature importances
    models = train_xgboost_model(X_train, y_train, args.out_dir, feature_names)

    # 运行SHAP可解释性分析
    # Run SHAP interpretability analysis
    run_final_shap_plots(models, X_test, feature_names, args.out_dir)

    print("\nScript finished successfully. SHAP analysis plots are saved.")


if __name__ == "__main__":
    main()
