import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
import time
import os
from itertools import cycle

# 设置结果保存路径
RESULTS_PATH = "/home/zqlibinyu/prediction/results1/"
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
        self.relu = nn.ReLU()

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
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU()

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(),
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
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )

        # MDFA模块 (需要将特征重塑为2D)
        self.mdfa = MDFA(dim_in=1, dim_out=16)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(16 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 初始特征处理
        x = self.input_layer(x)

        # 重塑为2D特征图 (32x32=1024)
        x = x.view(-1, 1, 32, 32)

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


def calculate_all_metrics(y_true, y_pred, y_proba=None, average='macro'):
    """计算所有评估指标"""
    metrics = {}

    # 基础指标
    metrics['Acc'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    # 计算灵敏度和特异性
    n_classes = len(np.unique(y_true))
    sensitivity = []
    specificity = []

    # 二分类或多分类
    if n_classes == 2:
        # 二分类情况
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    else:
        # 多分类情况
        cm = confusion_matrix(y_true, y_pred)
        for i in range(n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fp + fn)

            sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    metrics['Sen'] = np.mean(sensitivity)
    metrics['Spc'] = np.mean(specificity)

    # 计算AUROC（如果需要概率）
    if y_proba is not None:
        try:
            if n_classes == 2:
                metrics['AUROC'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # 多分类AUROC
                y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
                metrics['AUROC'] = roc_auc_score(y_true_bin, y_proba, average=average, multi_class='ovr')
        except:
            metrics['AUROC'] = 0.0

    return metrics


def train_pytorch_model(model_class, X_train, y_train, X_val=None, y_val=None):
    """训练PyTorch模型 - 使用AdamW优化器和标签平滑"""
    # 创建DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 如果有验证集，创建验证集DataLoader
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=64)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型初始化
    num_classes = len(np.unique(y_train))
    model = model_class(X_train.shape[1], num_classes).to(device)

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
    best_model_state = None

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

        # 验证阶段（如果有验证集）
        if val_loader is not None:
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for data, target in val_loader:
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

    # 加载最佳模型（如果有验证集）
    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 将模型移回CPU
    model = model.cpu()

    return model, best_acc if val_loader is not None else 0.0


def train_hist_gradient_boosting(X_train, y_train, X_val=None, y_val=None):
    """训练高效的梯度提升树模型"""
    print("训练HistGradientBoostingClassifier...")
    start_time = time.time()

    # 使用HistGradientBoosting - 更快更高效
    gb = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_leaf=20,
        max_bins=128,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=1
    )

    # 训练模型
    if X_val is not None and y_val is not None:
        # 如果有验证集，使用验证集进行早停
        gb.fit(X_train, y_train)
        acc_gb = accuracy_score(y_val, gb.predict(X_val))
    else:
        # 如果没有验证集，使用训练集评估
        gb.fit(X_train, y_train)
        acc_gb = accuracy_score(y_train, gb.predict(X_train))

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
    return np.argmax(avg_probs, axis=1), avg_probs


def create_optimized_ensemble(X_train, y_train, X_val=None, y_val=None):
    """创建优化的模型集成"""
    print("\n=== 创建优化的模型集成 ===")

    # 训练多个模型
    models = []
    weights = []
    accuracies = []

    # 1. 训练PyTorch模型 (带MDFA)
    print("训练PyTorch模型 (带MDFA)...")
    start_time = time.time()

    # 创建模型实例
    num_classes = len(np.unique(y_train))
    model_mdfa = AdvancedMLPWithMDFA(X_train.shape[1], num_classes)

    # 训练模型
    model_mdfa, acc_mdfa = train_pytorch_model(AdvancedMLPWithMDFA, X_train, y_train, X_val, y_val)

    models.append(model_mdfa)
    weights.append(0.5)
    accuracies.append(acc_mdfa)
    print(f"  MDFA模型准确率: {acc_mdfa:.4f}, 耗时: {time.time() - start_time:.2f}秒")

    # 2. 训练SVM
    print("训练SVM模型...")
    start_time = time.time()
    svm = SVC(probability=True, kernel='rbf', C=10, gamma='scale')
    svm.fit(X_train, y_train)
    if X_val is not None and y_val is not None:
        acc_svm = accuracy_score(y_val, svm.predict(X_val))
    else:
        acc_svm = accuracy_score(y_train, svm.predict(X_train))
    models.append(svm)
    weights.append(0.3)
    accuracies.append(acc_svm)
    print(f"  SVM模型准确率: {acc_svm:.4f}, 耗时: {time.time() - start_time:.2f}秒")

    # 3. 训练梯度提升树 (高效版)
    start_time = time.time()
    gb, acc_gb = train_hist_gradient_boosting(X_train, y_train, X_val, y_val)
    models.append(gb)
    weights.append(0.2)
    accuracies.append(acc_gb)

    # 加权平均预测
    if X_val is not None and y_val is not None:
        print("计算加权平均预测...")
        y_pred, y_proba = weighted_average_predict(models, weights, X_val)
        metrics = calculate_all_metrics(y_val, y_pred, y_proba)
        acc = metrics['Acc']
        print(f"集成模型准确率: {acc:.4f}")
        print(
            f"其他指标: Sen={metrics['Sen']:.4f}, Spc={metrics['Spc']:.4f}, MCC={metrics['MCC']:.4f}, AUROC={metrics['AUROC']:.4f}")
    else:
        acc = np.mean(accuracies)
        print(f"模型平均准确率: {acc:.4f}")

    return models, weights, acc


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


def save_roc_curve(y_true, y_proba, filename='roc_curve.png'):
    """保存ROC曲线到文件"""
    n_classes = len(np.unique(y_true))

    if n_classes == 2:
        # 二分类ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(RESULTS_PATH, filename))
        plt.close()
    else:
        # 多分类ROC曲线
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

        # 计算每个类别的ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 计算微平均ROC曲线
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # 绘制所有ROC曲线
        plt.figure(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red',
                        'purple', 'brown', 'pink', 'gray', 'olive'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(RESULTS_PATH, filename))
        plt.close()

    print(f"ROC曲线已保存为 {RESULTS_PATH}{filename}")


def save_detailed_training_summary(metrics, filename='detailed_training_summary.txt'):
    """保存详细训练摘要到文件"""
    with open(os.path.join(RESULTS_PATH, filename), 'w') as f:
        f.write("详细训练摘要\n")
        f.write("=" * 60 + "\n")

        # 基本信息
        f.write("\n[基本信息]\n")
        f.write("-" * 30 + "\n")
        f.write(f"数据样本数: {metrics['num_samples']}\n")
        f.write(f"特征数: {metrics['num_features']}\n")
        f.write(f"类别数: {len(metrics['class_distribution'])}\n")
        f.write(f"类别分布: {metrics['class_distribution']}\n")

        # 性能指标
        f.write("\n[性能指标]\n")
        f.write("-" * 30 + "\n")
        f.write(f"准确率 (Accuracy): {metrics['Acc']:.4f}\n")
        f.write(f"精确率 (Precision): {metrics['Precision']:.4f}\n")
        f.write(f"召回率 (Recall): {metrics['Recall']:.4f}\n")
        f.write(f"F1分数: {metrics['F1']:.4f}\n")
        f.write(f"灵敏度 (Sensitivity): {metrics['Sen']:.4f}\n")
        f.write(f"特异性 (Specificity): {metrics['Spc']:.4f}\n")
        f.write(f"马修斯相关系数 (MCC): {metrics['MCC']:.4f}\n")
        f.write(f"AUROC: {metrics['AUROC']:.4f}\n")

        # 交叉验证结果
        if 'cv_mean_accuracy' in metrics:
            f.write("\n[交叉验证结果]\n")
            f.write("-" * 30 + "\n")
            f.write(f"平均准确率: {metrics['cv_mean_accuracy']:.4f}\n")
            f.write(f"标准差: {metrics['cv_std_accuracy']:.4f}\n")
            f.write(f"各折准确率: {metrics['cv_fold_accuracies']}\n")

        # 训练信息
        if 'training_time' in metrics:
            f.write("\n[训练信息]\n")
            f.write("-" * 30 + "\n")
            f.write(f"总训练时间: {metrics['training_time']:.2f}秒\n")
            if 'epochs' in metrics:
                f.write(f"训练轮数: {metrics['epochs']}\n")
            if 'batch_size' in metrics:
                f.write(f"批次大小: {metrics['batch_size']}\n")

        # 模型信息
        if 'model_params' in metrics:
            f.write("\n[模型信息]\n")
            f.write("-" * 30 + "\n")
            for key, value in metrics['model_params'].items():
                f.write(f"{key}: {value}\n")

        # 特征重要性（如果有）
        if 'top_features' in metrics:
            f.write("\n[重要特征]\n")
            f.write("-" * 30 + "\n")
            for i, (feature, importance) in enumerate(metrics['top_features'], 1):
                f.write(f"{i}. {feature}: {importance:.4f}\n")

    print(f"详细训练摘要已保存为 {RESULTS_PATH}{filename}")


def save_metrics_table(metrics, filename='metrics_table.csv'):
    """保存指标表格到CSV文件"""
    # 创建指标数据的DataFrame
    metrics_data = {
        'Metric': ['准确率', '精确率', '召回率', 'F1分数', '灵敏度', '特异性', 'MCC', 'AUROC'],
        'Value': [
            metrics.get('Acc', 0),
            metrics.get('Precision', 0),
            metrics.get('Recall', 0),
            metrics.get('F1', 0),
            metrics.get('Sen', 0),
            metrics.get('Spc', 0),
            metrics.get('MCC', 0),
            metrics.get('AUROC', 0)
        ]
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(RESULTS_PATH, filename), index=False)
    print(f"指标表格已保存为 {RESULTS_PATH}{filename}")


def cross_validate_ensemble(X, y, n_splits=5):
    """执行5折交叉验证"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_metrics = []
    best_ensemble = None
    best_acc = 0.0

    print(f"\n开始{n_splits}折交叉验证...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== 折 {fold + 1}/{n_splits} ===")

        # 分割数据
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 创建优化的模型集成
        models, weights, acc = create_optimized_ensemble(X_train, y_train, X_val, y_val)

        # 计算所有指标
        y_pred, y_proba = weighted_average_predict(models, weights, X_val)
        metrics = calculate_all_metrics(y_val, y_pred, y_proba)

        # 记录准确率和指标
        fold_accuracies.append(acc)
        fold_metrics.append(metrics)

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            best_ensemble = {
                'models': models,
                'weights': weights,
                'accuracy': acc,
                'metrics': metrics,
                'model_state_dict': models[0].state_dict() if isinstance(models[0], nn.Module) else None
            }
            print(f"新的最佳模型 (准确率: {acc:.4f})")

    # 计算平均准确率和指标
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    # 计算平均指标
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in fold_metrics])

    print("\n" + "=" * 60)
    print(f"{n_splits}折交叉验证结果:")
    print(f"各折准确率: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")

    # 打印所有指标的平均值
    print("\n平均指标:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")

    return best_ensemble, fold_accuracies, mean_acc, std_acc, avg_metrics


def save_feature_importance_plot(models, feature_names, top_n=20, filename='feature_importance.png'):
    """保存特征重要性图"""
    try:
        # 尝试从梯度提升树获取特征重要性
        if hasattr(models[2], 'feature_importances_'):
            feature_importance = models[2].feature_importances_

            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False).head(top_n)

            # 绘制特征重要性图
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
            plt.title(f'Top {top_n} 特征重要性')
            plt.xlabel('重要性得分')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_PATH, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"特征重要性图已保存为 {RESULTS_PATH}{filename}")

            # 保存特征重要性表格
            importance_df.to_csv(os.path.join(RESULTS_PATH, 'feature_importance.csv'), index=False)
            print(f"特征重要性表格已保存为 {RESULTS_PATH}feature_importance.csv")

            return importance_df
    except Exception as e:
        print(f"无法保存特征重要性图: {e}")
        return None


def save_training_curves(history, filename='training_curves.png'):
    """保存训练曲线（如果可用）"""
    if history and 'loss' in history and 'val_loss' in history:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='训练准确率')
        plt.plot(history['val_accuracy'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存为 {RESULTS_PATH}{filename}")


def save_training_summary(metrics, filename='training_summary.txt'):
    """保存训练摘要到文件"""
    with open(os.path.join(RESULTS_PATH, filename), 'w') as f:
        f.write("训练摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"数据样本数: {metrics['num_samples']}\n")
        f.write(f"特征数: {metrics['num_features']}\n")
        f.write(f"类别分布: {metrics['class_distribution']}\n")
        f.write(f"准确率 (Acc): {metrics['Acc']:.4f}\n")
        f.write(f"灵敏度 (Sen): {metrics['Sen']:.4f}\n")
        f.write(f"特异性 (Spc): {metrics['Spc']:.4f}\n")
        f.write(f"马修斯相关系数 (MCC): {metrics['MCC']:.4f}\n")
        f.write(f"AUROC: {metrics['AUROC']:.4f}\n")
        f.write(f"精确率 (Precision): {metrics['Precision']:.4f}\n")
        f.write(f"召回率 (Recall): {metrics['Recall']:.4f}\n")
        f.write(f"F1分数: {metrics['F1']:.4f}\n")

        # 如果有交叉验证结果，也保存
        if 'cv_mean_accuracy' in metrics:
            f.write(f"交叉验证平均准确率: {metrics['cv_mean_accuracy']:.4f}\n")
            f.write(f"交叉验证标准差: {metrics['cv_std_accuracy']:.4f}\n")

        # 如果有训练时间信息
        if 'training_time' in metrics:
            f.write(f"训练时间: {metrics['training_time']:.2f}秒\n")

        # 如果有模型参数信息
        if 'model_params' in metrics:
            f.write(f"模型参数: {metrics['model_params']}\n")

    print(f"训练摘要已保存为 {RESULTS_PATH}{filename}")

def main():
    """主函数"""
    try:
        # 1. 加载数据
        filepath = "/home/zqlibinyu/prediction/data/output/features_test_sheet1.csv"
        df = pd.read_csv(filepath)
        X = df.drop(columns=["label"]).values
        y = df["label"].values
        feature_names = df.drop(columns=["label"]).columns.tolist()

        print(f"数据加载完成: {X.shape[0]}个样本, {X.shape[1]}个特征")
        print(f"类别分布: {np.bincount(y)}")

        # 2. 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 3. 执行5折交叉验证
        start_time = time.time()
        best_ensemble, fold_accuracies, mean_acc, std_acc, avg_metrics = cross_validate_ensemble(X_scaled, y,
                                                                                                 n_splits=5)
        training_time = time.time() - start_time

        # 4. 使用最佳集成模型进行最终评估
        models = best_ensemble['models']
        weights = best_ensemble['weights']
        best_acc = best_ensemble['accuracy']

        # 确保模型已加载权重
        if 'model_state_dict' in best_ensemble and best_ensemble['model_state_dict'] is not None:
            if isinstance(models[0], nn.Module):
                models[0].load_state_dict(best_ensemble['model_state_dict'])

        # 在完整数据集上评估
        y_pred_full, y_proba_full = weighted_average_predict(models, weights, X_scaled)
        final_metrics = calculate_all_metrics(y, y_pred_full, y_proba_full)

        print("\n" + "=" * 60)
        print("最终测试结果（在整个数据集上）")
        print("=" * 60)
        print(f"测试准确率: {final_metrics['Acc']:.4f}")
        print(f"灵敏度 (Sen): {final_metrics['Sen']:.4f}")
        print(f"特异性 (Spc): {final_metrics['Spc']:.4f}")
        print(f"马修斯相关系数 (MCC): {final_metrics['MCC']:.4f}")
        print(f"AUROC: {final_metrics['AUROC']:.4f}")
        print(f"精确率: {final_metrics['Precision']:.4f}")
        print(f"召回率: {final_metrics['Recall']:.4f}")
        print(f"F1分数: {final_metrics['F1']:.4f}")

        # 5. 保存模型
        model_dict = {
            'models': models,
            'weights': weights,
            'scaler': scaler,
            'accuracy': final_metrics['Acc'],
            'metrics': final_metrics,
            'feature_names': feature_names,
            'cv_mean_accuracy': mean_acc,
            'cv_std_accuracy': std_acc,
            'cv_avg_metrics': avg_metrics,
            'training_time': training_time
        }

        # 根据准确率选择模型文件名
        if final_metrics['Acc'] >= 0.96:
            model_filename = 'high_accuracy_mdfa_model.pkl'
        else:
            model_filename = 'accuracy_mdfa_model.pkl'

        joblib.dump(model_dict, os.path.join(RESULTS_PATH, model_filename))
        print(f"\n模型已保存为: {RESULTS_PATH}{model_filename}")

        # 6. 保存各种报告和图表
        print("\n正在生成报告和图表...")

        # 分类报告
        save_classification_report(y, y_pred_full)

        # 混淆矩阵
        save_confusion_matrix(y, y_pred_full)

        # ROC曲线
        save_roc_curve(y, y_proba_full)

        # 训练摘要
        summary_metrics = {
            'num_samples': X.shape[0],
            'num_features': X.shape[1],
            'class_distribution': np.bincount(y),
            'Acc': final_metrics['Acc'],
            'Sen': final_metrics['Sen'],
            'Spc': final_metrics['Spc'],
            'MCC': final_metrics['MCC'],
            'AUROC': final_metrics['AUROC'],
            'Precision': final_metrics['Precision'],
            'Recall': final_metrics['Recall'],
            'F1': final_metrics['F1'],
            'cv_mean_accuracy': mean_acc,
            'cv_std_accuracy': std_acc,
            'training_time': training_time
        }
        save_training_summary(summary_metrics)

        # 指标表格
        save_metrics_table(final_metrics)

        # 特征重要性图
        save_feature_importance_plot(models, feature_names)

        # 7. 保存预测结果
        results_df = pd.DataFrame({
            'true_label': y,
            'predicted_label': y_pred_full,
            'prediction_correct': y == y_pred_full
        })
        results_df.to_csv(os.path.join(RESULTS_PATH, 'prediction_results.csv'), index=False)
        print(f"预测结果已保存为 {RESULTS_PATH}prediction_results.csv")

        # 8. 保存概率预测结果
        proba_df = pd.DataFrame(y_proba_full, columns=[f'class_{i}_probability' for i in range(y_proba_full.shape[1])])
        proba_df['true_label'] = y
        proba_df['predicted_label'] = y_pred_full
        proba_df.to_csv(os.path.join(RESULTS_PATH, 'prediction_probabilities.csv'), index=False)
        print(f"预测概率已保存为 {RESULTS_PATH}prediction_probabilities.csv")

        # 9. 生成最终性能报告
        print("\n" + "=" * 60)
        print("最终性能报告")
        print("=" * 60)
        print(f"总训练时间: {training_time:.2f}秒")
        print(f"交叉验证平均准确率: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"最终测试准确率: {final_metrics['Acc']:.4f}")
        print(f"模型性能评估完成!")

        # 10. 保存完整的实验配置
        config = {
            'data_path': filepath,
            'feature_count': X.shape[1],
            'sample_count': X.shape[0],
            'class_distribution': np.bincount(y).tolist(),
            'cv_folds': 5,
            'model_weights': weights,
            'final_accuracy': final_metrics['Acc'],
            'training_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        config_df = pd.DataFrame([config])
        config_df.to_csv(os.path.join(RESULTS_PATH, 'experiment_config.csv'), index=False)
        print(f"实验配置已保存为 {RESULTS_PATH}experiment_config.csv")

        print(f"\n所有结果已保存到: {RESULTS_PATH}")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()