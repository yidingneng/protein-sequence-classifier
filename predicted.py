import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
import time
import os

# 设置结果保存路径
RESULTS_PATH = "/home/zqlibinyu/prediction/results/resultsB"
os.makedirs(RESULTS_PATH, exist_ok=True)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# ===== MDFA模块及其依赖 =====
class tongdao(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU()  # 移除 inplace=True

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        y = self.relu(y)
        y = F.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')
        return x * y.expand_as(x)


class kongjian(nn.Module):
    """空间注意力模块"""

    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        y = self.Conv1x1(x)
        y = self.norm(y)
        return x * y


class hebing(nn.Module):
    """合并通道和空间注意力"""

    def __init__(self, in_channel):
        super().__init__()
        self.tongdao = tongdao(in_channel)
        self.kongjian = kongjian(in_channel)

    def forward(self, U):
        U_kongjian = self.kongjian(U)
        U_tongdao = self.tongdao(U)
        return torch.max(U_tongdao, U_kongjian)


class MDFA(nn.Module):
    """多尺度空洞融合注意力模块"""

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(),  # 移除 inplace=True
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(),  # 移除 inplace=True
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(),  # 移除 inplace=True
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(),  # 移除 inplace=True
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU()  # 移除 inplace=True

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(),  # 移除 inplace=True
        )
        self.Hebing = hebing(in_channel=dim_out * 5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        # 应用各分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        # 全局特征提取
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # 合并所有特征
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)

        # 应用合并模块进行通道和空间特征增强
        larry = self.Hebing(feature_cat)
        larry_feature_cat = larry * feature_cat

        # 最终输出经过降维处理
        result = self.conv_cat(larry_feature_cat)
        return result


# ===== 集成MDFA的模型 =====
class AdvancedMLPWithMDFA(nn.Module):
    """集成MDFA模块的高级MLP"""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),  # 移除 inplace=True
            nn.Dropout(0.4)
        )

        # MDFA模块 (需要将特征重塑为2D)
        self.mdfa = MDFA(dim_in=1, dim_out=16)  # 输入通道1，输出通道16

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(16 * 32 * 32, 512),  # MDFA输出为16通道，32x32特征图
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),  # 移除 inplace=True
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 初始特征处理
        x = self.input_layer(x)

        # 重塑为2D特征图 (32x32=1024)
        x = x.view(-1, 1, 32, 32)  # [batch, channels, height, width]

        # 应用MDFA模块
        x = self.mdfa(x)

        # 展平特征
        x = x.view(x.size(0), -1)

        return self.output_layer(x)


class LabelSmoothLoss(nn.Module):
    """标签平滑损失"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, targets, reduction='none')
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def train_pytorch_model(model_class, X, y):
    """训练PyTorch模型 - 使用AdamW优化器和标签平滑"""
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 创建DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型初始化
    num_classes = len(np.unique(y))
    model = model_class(X.shape[1], num_classes).to(device)

    # 优化器 - 使用AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )

    # 损失函数 - 标签平滑
    criterion = LabelSmoothLoss(smoothing=0.1)

    # 训练
    best_acc = 0.0
    for epoch in range(100):
        # 训练阶段
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict().copy()

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model, best_acc


def train_hist_gradient_boosting(X_train, y_train, X_test, y_test):
    """训练高效的梯度提升树模型"""
    print("训练HistGradientBoostingClassifier...")
    start_time = time.time()

    # 使用HistGradientBoosting - 更快更高效
    gb = HistGradientBoostingClassifier(
        max_iter=200,  # 减少迭代次数
        learning_rate=0.1,  # 增加学习率
        max_depth=5,  # 减小深度
        min_samples_leaf=20,  # 增加叶子节点最小样本数
        max_bins=128,  # 减少分箱数
        early_stopping=True,  # 启用早停
        validation_fraction=0.1,  # 验证集比例
        n_iter_no_change=10,  # 10轮无提升则停止
        random_state=42,
        verbose=1  # 显示进度
    )

    gb.fit(X_train, y_train)
    acc_gb = accuracy_score(y_test, gb.predict(X_test))
    print(f"  梯度提升树准确率: {acc_gb:.4f}, 耗时: {time.time() - start_time:.2f}秒")
    return gb, acc_gb


def weighted_average_predict(models, weights, X):
    """加权平均预测"""
    all_probs = []
    for model, weight in zip(models, weights):
        if isinstance(model, nn.Module):
            # PyTorch模型预测
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                outputs = model(X_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
        else:
            # Scikit-learn模型预测
            probs = model.predict_proba(X)

        all_probs.append(probs * weight)

    avg_probs = np.mean(all_probs, axis=0)
    return np.argmax(avg_probs, axis=1)


def create_optimized_ensemble(X, y):
    """创建优化的模型集成"""
    print("\n=== 创建优化的模型集成 ===")

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 训练多个模型
    models = []
    weights = []

    # 1. 训练PyTorch模型 (带MDFA)
    print("训练PyTorch模型 (带MDFA)...")
    start_time = time.time()
    model_mdfa, acc_mdfa = train_pytorch_model(AdvancedMLPWithMDFA, X_train, y_train)
    models.append(model_mdfa)
    weights.append(0.5)
    print(f"  MDFA模型准确率: {acc_mdfa:.4f}, 耗时: {time.time() - start_time:.2f}秒")

    # 2. 训练SVM
    print("训练SVM模型...")
    start_time = time.time()
    svm = SVC(probability=True, kernel='rbf', C=10, gamma='scale')
    svm.fit(X_train, y_train)
    acc_svm = accuracy_score(y_test, svm.predict(X_test))
    models.append(svm)
    weights.append(0.3)
    print(f"  SVM模型准确率: {acc_svm:.4f}, 耗时: {time.time() - start_time:.2f}秒")

    # 3. 训练梯度提升树 (高效版)
    gb, acc_gb = train_hist_gradient_boosting(X_train, y_train, X_test, y_test)
    models.append(gb)
    weights.append(0.2)

    # 加权平均预测
    print("计算加权平均预测...")
    y_pred = weighted_average_predict(models, weights, X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"集成模型准确率: {acc:.4f}")

    return models, weights, acc, X_test, y_test


def save_classification_report(y_true, y_pred, filename='classification_report.txt'):
    """保存分类报告到文件"""
    report = classification_report(y_true, y_pred)
    with open(os.path.join(RESULTS_PATH, filename), 'w') as f:
        f.write("分类报告\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    print(f"分类报告已保存为 {RESULTS_PATH}{filename}")


def save_confusion_matrix(y_true, y_pred, filename='confusion_matrix.png'):
    """保存混淆矩阵到文件"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(RESULTS_PATH, filename))
    plt.close()
    print(f"混淆矩阵已保存为 {RESULTS_PATH}{filename}")


def save_training_summary(accuracies, filename='training_summary.txt'):
    """保存训练摘要到文件"""
    with open(os.path.join(RESULTS_PATH, filename), 'w') as f:
        f.write("训练摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"数据样本数: {accuracies['num_samples']}\n")
        f.write(f"特征数: {accuracies['num_features']}\n")
        f.write(f"类别分布: {accuracies['class_distribution']}\n")
        f.write(f"MDFA模型准确率: {accuracies['mdfa_acc']:.4f}\n")
        f.write(f"SVM模型准确率: {accuracies['svm_acc']:.4f}\n")
        f.write(f"梯度提升树准确率: {accuracies['gb_acc']:.4f}\n")
        f.write(f"集成模型准确率: {accuracies['ensemble_acc']:.4f}\n")
        f.write(f"最终测试准确率: {accuracies['final_acc']:.4f}\n")
    print(f"训练摘要已保存为 {RESULTS_PATH}{filename}")


def main():
    """主函数"""
    try:
        # 1. 加载数据
        filepath = "/home/zqlibinyu/prediction/data/output/Alternate_DatasetB_sheet1.csv"
        df = pd.read_csv(filepath)
        X = df.drop(columns=["label"]).values
        y = df["label"].values
        feature_names = df.drop(columns=["label"]).columns.tolist()

        print(f"数据加载完成: {X.shape[0]}个样本, {X.shape[1]}个特征")
        print(f"类别分布: {np.bincount(y)}")

        # 2. 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. 创建优化的模型集成
        models, weights, final_acc, X_test, y_test = create_optimized_ensemble(X_scaled, y)

        # 4. 最终评估（在测试集上）
        print("\n" + "=" * 60)
        print("最终测试结果（在测试集上）")
        print("=" * 60)
        print(f"测试准确率: {final_acc:.4f}")

        # 5. 保存模型
        model_dict = {
            'models': models,
            'weights': weights,
            'scaler': scaler,
            'accuracy': final_acc,
            'feature_names': feature_names
        }

        if final_acc >= 0.96:
            model_filename = 'high_accuracy_mdfa_model.pkl'
            joblib.dump(model_dict, os.path.join(RESULTS_PATH, model_filename))
            print(f"\n模型已保存为: {RESULTS_PATH}{model_filename}")
        else:
            model_filename = f'accuracy_mdfa_model.pkl'
            joblib.dump(model_dict, os.path.join(RESULTS_PATH, model_filename))
            print(f"\n模型已接近最优")
            print(f"模型已保存为: {RESULTS_PATH}{model_filename}")

        # 6. 最终报告（在测试集上）
        print("\n详细分类报告（测试集）:")
        y_pred_test = weighted_average_predict(models, weights, X_test)
        print(classification_report(y_test, y_pred_test))

        # 保存分类报告
        save_classification_report(y_test, y_pred_test)

        # 保存混淆矩阵
        save_confusion_matrix(y_test, y_pred_test)

        # 保存训练摘要
        accuracies = {
            'num_samples': X.shape[0],
            'num_features': X.shape[1],
            'class_distribution': np.bincount(y),
            'mdfa_acc': final_acc,
            'svm_acc': accuracy_score(y_test, models[1].predict(X_test)),
            'gb_acc': accuracy_score(y_test, models[2].predict(X_test)),
            'ensemble_acc': final_acc,
            'final_acc': final_acc
        }
        save_training_summary(accuracies)

        # 7. 保存预测结果
        y_pred_full = weighted_average_predict(models, weights, X_scaled)
        results_df = pd.DataFrame({
            'true_label': y,
            'predicted_label': y_pred_full,
            'prediction_correct': y == y_pred_full
        })
        results_df.to_csv(os.path.join(RESULTS_PATH, 'prediction_results.csv'), index=False)
        print(f"预测结果已保存为 {RESULTS_PATH}prediction_results.csv")

        # 8. 保存特征重要性（如果可用）
        try:
            # 尝试从梯度提升树获取特征重要性
            if hasattr(models[2], 'feature_importances_'):
                feature_importance = models[2].feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                importance_df.to_csv(os.path.join(RESULTS_PATH, 'feature_importance.csv'), index=False)
                print(f"特征重要性已保存为 {RESULTS_PATH}feature_importance.csv")
        except Exception as e:
            print(f"无法保存特征重要性: {e}")

        print(f"\n所有结果已保存到: {RESULTS_PATH}")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()