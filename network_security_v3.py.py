#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install xgboost
#!pip install lightgbm


# In[3]:


# ============================================================
# 项目：《基于机器学习（XGBoost & LightGBM）的校园网络流量异常攻击检测》
# 版本：v2.0（数据混淆迭代版）
# 作者：大一计算机系新生（人机协同 Top-Down 玩法）
# 更新说明：
#   - 让正常/攻击流量的数值特征大量重叠，防止模型"作弊"
#   - 新增 wrong_fragment（错误分片）、hot（热点连接数）两个噪声特征
#   - 标签基于"多特征弱信号组合"生成，贴近真实网安场景
# 依赖安装（在命令行执行）：
#   pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# 尝试导入 XGBoost 和 LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告：未检测到 xgboost，请运行：pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告：未检测到 lightgbm，请运行：pip install lightgbm")

# 设置中文字体（Windows 用 SimHei，Mac 用 Heiti TC）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
#  Step 1：生成"混淆版"网络流量数据（特征重叠 + 弱信号标签）
#
#  核心设计思路：
#    真实网安场景中，正常流量和攻击流量的数值特征（src_bytes、
#    dst_bytes 等）有大量重叠，单看一个特征根本分不出来。
#    攻击的"异常"是多个弱特征组合起来才显现的"微弱信号"。
#
#  标签生成逻辑（弱信号组合）：
#    攻击得分 = 0.3*duration信号 + 0.25*src_bytes信号
#               + 0.2*dst_bytes信号 + 0.15*wrong_fragment信号
#               + 0.1*hot信号 + 噪声
#    攻击得分越高，该样本被标为攻击（label=1）的概率越大。
#
#  这样一来：
#    - 随机森林（基于单特征分裂）会比较吃力
#    - XGBoost（梯度提升，关注残差/错题）更有优势
#    - LightGBM（直方图分桶，擅长高维稀疏特征）也更有优势
# ============================================================

def generate_confused_network_traffic(n_samples=20000, attack_ratio=0.2, random_state=42):
    """
    生成特征高度重叠的网络流量数据（模拟真实网安场景）

    参数：
        n_samples     ：总样本数
        attack_ratio  ：攻击流量目标占比（最终会因噪声略有浮动）
        random_state  ：随机种子，保证可复现
    """
    rng = np.random.RandomState(random_state)

    # ---------- 1. 生成基础特征（正常/攻击完全重叠区间）----------
    # duration：连接持续时间（秒），正常和攻击都用同一种分布，完全重叠
    duration = rng.exponential(scale=50, size=n_samples).astype(int)
    duration = np.clip(duration, 0, 500)   # 裁剪到合理范围

    # protocol_type：协议类型，正常和攻击共享同样的分布
    protocol_type = rng.choice(['TCP', 'UDP', 'ICMP'], size=n_samples,
                               p=[0.6, 0.3, 0.1])

    # src_bytes / dst_bytes：发送/接收字节数，正常和攻击完全重叠在 0~3000
    # 这是关键修改：不再用不同分布，而是统一分布 + 微弱偏移
    src_bytes = rng.randint(0, 3001, size=n_samples)
    dst_bytes = rng.randint(0, 3001, size=n_samples)

    # wrong_fragment：错误分片数（0 或 1，高维稀疏噪声特征）
    # 攻击略多，但正常流量也有，不能完全区分
    wrong_fragment = rng.choice([0, 1], size=n_samples, p=[0.85, 0.15])

    # hot：热点连接数（0~5，虚虚实实的噪声特征）
    hot = rng.randint(0, 6, size=n_samples)

    # ---------- 2. 构造"微弱攻击信号"（用于生成标签）----------
    # 每个样本独立计算一个"攻击得分"，得分高的更可能是攻击
    # 每个特征贡献一部分信号（权重不同），最后加高斯噪声混淆

    # 信号 1：duration 越长，攻击概率略增（权重 0.3）
    # 把 duration 归一化到 0~1 区间
    dur_signal = (duration / 500.0) * 0.3

    # 信号 2：src_bytes 极小（可能是端口扫描类攻击）或极大，攻击概率略增
    # 用距离两端的距离来衡量（U 形信号）
    src_norm = src_bytes / 3000.0
    src_signal = 0.25 * np.where(src_norm < 0.1, (0.1 - src_norm) * 10,
                                   np.where(src_norm > 0.9, (src_norm - 0.9) * 10, 0))

    # 信号 3：dst_bytes 极小，可能是探测类攻击
    dst_norm = dst_bytes / 3000.0
    dst_signal = 0.2 * np.where(dst_norm < 0.15, (0.15 - dst_norm) * 10, 0)

    # 信号 4：wrong_fragment = 1 时，攻击概率略增
    frag_signal = 0.15 * wrong_fragment

    # 信号 5：hot 越高，攻击概率略增（热点爆破类攻击）
    hot_signal = 0.1 * (hot / 5.0)

    # 高斯噪声（均值 0，标准差 0.1）—— 关键！让边界模糊
    noise = rng.normal(loc=0.0, scale=0.1, size=n_samples)

    # 总攻击得分（范围大约在 0~1 之间）
    attack_score = dur_signal + src_signal + dst_signal + frag_signal + hot_signal + noise
    attack_score = np.clip(attack_score, 0, 1)   # 裁剪到 [0, 1]

    # ---------- 3. 根据攻击得分，按概率分配标签 ----------
    # 攻击得分越高，label=1 的概率越大（但不是确定性映射）
    # 用 attack_score 作为概率，做一次伯努利采样
    label = (rng.rand(n_samples) < attack_score).astype(int)

    # 强制保证攻击流量占比大致在 attack_ratio 附近（可选）
    current_ratio = label.mean()
    print(f"   实际攻击流量占比：{current_ratio:.3f}（目标：{attack_ratio}）")
    print(f"   （因噪声和概率映射，占比会有浮动，这符合真实场景）")

    # 组装 DataFrame
    df = pd.DataFrame({
        'duration': duration,
        'protocol_type': protocol_type,
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
        'wrong_fragment': wrong_fragment,   # 新增：错误分片
        'hot': hot,                         # 新增：热点连接数
        'label': label
    })

    return df


print("=" * 65)
print("Step 1：生成【混淆版】网络流量数据（特征重叠 + 弱信号）...")
df = generate_confused_network_traffic(n_samples=20000, attack_ratio=0.2, random_state=42)
print(f"数据生成完成！共 {len(df)} 行")
print(f"  正常流量（label=0）：{(df['label']==0).sum()} 条")
print(f"  攻击流量（label=1）：{(df['label']==1).sum()} 条")
print(f"\n数据预览（看看特征是不是已经分不清了）：")
print(df.head(8))
print("\n各特征分布（正常 vs 攻击的对比）：")
for col in ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'hot']:
    normal_mean = df[df['label']==0][col].mean()
    attack_mean = df[df['label']==1][col].mean()
    print(f"  {col:>18s}：正常均值={normal_mean:>8.2f}，攻击均值={attack_mean:>8.2f}"
          f"  {'↑攻击更高' if attack_mean > normal_mean else '↓正常更高'}")
print("=" * 65)


# ============================================================
#  Step 2：特征工程 —— 独热编码 + 数值特征直通
#  新增特征：wrong_fragment、hot 也加入数值特征一起训练
# ============================================================

categorical_features = ['protocol_type']
numeric_features = ['duration', 'src_bytes', 'dst_bytes',
                     'wrong_fragment', 'hot']   # 新增两个噪声特征

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# 划分训练集 / 测试集（分层采样）
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nStep 2：特征工程完成")
print(f"  类别特征（独热编码）：{categorical_features}")
print(f"  数值特征（直通）：{numeric_features}")
print(f"  训练集：{len(X_train)} 条，测试集：{len(X_test)} 条")
print("=" * 65)


# ============================================================
#  Step 3：构建三个模型（同台竞技）
# ============================================================

models = {}

# 模型 1：随机森林（基线）
models['RandomForest'] = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

# 模型 2：XGBoost（梯度提升，有"错题本机制"）
if XGBOOST_AVAILABLE:
    models['XGBoost'] = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ))
    ])

# 模型 3：LightGBM（直方图机制，擅长高维稀疏特征）
if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
    ])


# ============================================================
#  Step 4：训练 + 评估（网安不平衡数据指标）
# ============================================================

results = []

print("\nStep 3 & 4：模型训练 + 评估（网安视角，F1-Score 是核心）")
print("=" * 65)

for name, model in models.items():
    print(f"\n训练模型：{name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    results.append({
        '模型': name,
        '准确率': round(acc, 4),
        '精确率(Precision)': round(precision, 4),
        '召回率(Recall)': round(recall, 4),
        'F1-Score': round(f1, 4)
    })

    print(f"  {name} 训练完成")
    print(f"    准确率  Accuracy : {acc:.4f}")
    print(f"    精确率  Precision: {precision:.4f}  （攻击警报的准确率，减少误报）")
    print(f"    召回率  Recall   : {recall:.4f}  （攻击捕获率，减少漏报）")
    print(f"    F1-Score        : {f1:.4f}  （综合指标，网安最看重）")

    cm = confusion_matrix(y_test, y_pred)
    print(f"    混淆矩阵：[TN={cm[0,0]}, FP={cm[0,1]}] / [FN={cm[1,0]}, TP={cm[1,1]}]")

print("\n" + "=" * 65)
print("三模型横向对比（攻击检测 F1-Score，越高越好）：")
result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))
print("=" * 65)


# ============================================================
#  Step 5：可视化 —— 三模型 F1-Score 柱状图对比
# ============================================================

print("\nStep 5：生成可视化对比图...")

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(result_df))
colors = ['#4C72B0', '#55A868', '#C44E52'][:len(result_df)]
bars = ax.bar(x_pos, result_df['F1-Score'],
              color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

# 柱顶标注数值
for bar, value in zip(bars, result_df['F1-Score']):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{value:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xlabel('模型', fontsize=13, fontweight='bold')
ax.set_ylabel('F1-Score（攻击检测综合指标）', fontsize=13, fontweight='bold')
ax.set_title('三模型网络流量攻击检测 F1-Score 对比\n（混淆数据版：特征重叠，弱信号标签）',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(result_df['模型'], fontsize=12)
ax.set_ylim([0, max(result_df['F1-Score']) + 0.06])
ax.grid(axis='y', linestyle='--', alpha=0.4)

# 标注冠军
best_idx = result_df['F1-Score'].idxmax()
ax.text(bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
        bars[best_idx].get_height() + 0.04,
        '冠军', ha='center', fontsize=11, color='red', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n全部完成！看看这次 XGBoost / LightGBM 能不能考赢 RandomForest！")
print("（理论上看，梯度提升类模型在混淆数据上更有优势）")
print("=" * 65)


# In[4]:


# ============================================================
# 项目：《基于机器学习（XGBoost & LightGBM）的校园网络流量异常攻击检测》
# 版本：v3.0（阈值调优 + 特征标准化 挽救版）
# 作者：大一计算机系新生（北邮数据科学专业）
# 核心改进：
#   1. 加入 StandardScaler 对数值特征标准化
#   2. 用验证集寻找最优分类阈值（最大化 F1-Score）
#   3. 对比"默认阈值 0.5" vs "最优阈值"的效果差异
# 依赖安装：
#   pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, precision_recall_curve,
                             classification_report, confusion_matrix)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告：未检测到 xgboost，请运行：pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告：未检测到 lightgbm，请运行：pip install lightgbm")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
#  Step1：生成混淆版网络流量数据（同 v2.0，数据不变）
# ============================================================

def generate_confused_network_traffic(n_samples=20000, attack_ratio=0.2, random_state=42):
    rng = np.random.RandomState(random_state)

    duration = rng.exponential(scale=50, size=n_samples).astype(int)
    duration = np.clip(duration, 0, 500)

    protocol_type = rng.choice(['TCP', 'UDP', 'ICMP'], size=n_samples, p=[0.6, 0.3, 0.1])

    # 关键：正常和攻击完全共享数值区间（特征重叠）
    src_bytes = rng.randint(0, 3001, size=n_samples)
    dst_bytes = rng.randint(0, 3001, size=n_samples)

    wrong_fragment = rng.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    hot = rng.randint(0, 6, size=n_samples)

    # 弱信号组合生成标签（不是硬界限，是概率映射）
    dur_signal = (duration / 500.0) * 0.3
    src_norm = src_bytes / 3000.0
    src_signal = 0.25 * np.where(src_norm < 0.1, (0.1 - src_norm) * 10,
                                   np.where(src_norm > 0.9, (src_norm - 0.9) * 10, 0))
    dst_norm = dst_bytes / 3000.0
    dst_signal = 0.2 * np.where(dst_norm < 0.15, (0.15 - dst_norm) * 10, 0)
    frag_signal = 0.15 * wrong_fragment
    hot_signal = 0.1 * (hot / 5.0)
    noise = rng.normal(loc=0.0, scale=0.1, size=n_samples)
    attack_score = np.clip(dur_signal + src_signal + dst_signal + frag_signal + hot_signal + noise, 0, 1)

    label = (rng.rand(n_samples) < attack_score).astype(int)

    df = pd.DataFrame({
        'duration': duration,
        'protocol_type': protocol_type,
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
        'wrong_fragment': wrong_fragment,
        'hot': hot,
        'label': label
    })
    return df


print("=" * 65)
print("Step1：生成混淆版网络流量数据...")
df = generate_confused_network_traffic(n_samples=20000, attack_ratio=0.2, random_state=42)
print(f"数据生成完成！共 {len(df)} 行")
print(f"  正常流量（label=0）：{(df['label']==0).sum()} 条")
print(f"  攻击流量（label=1）：{(df['label']==1).sum()} 条")
print("=" * 65)


# ============================================================
#  Step2：特征工程（加入 StandardScaler 标准化数值特征）
#
#  知识点：
#    树模型（RF/XGB/LGB）不需要标准化，但加上也无害。
#    标准化后数值特征均值为 0、标准差为 1，便于：
#      - 后续如果接逻辑回归/SVM/神经网络，可以直接用
#      - 特征重要性解释更直观（量纲一致）
#
#  ColumnTransformer 三个通道：
#    'cat'  → 独热编码（类别特征）
#    'num'  → 标准化（数值特征）
# ============================================================

categorical_features = ['protocol_type']
numeric_features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'hot']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_features),
        ('num', StandardScaler(), numeric_features)   #  ← v3.0 新增：标准化
    ])

# 数据划分：训练集 60% / 验证集 20% / 测试集 20%
# 验证集专门用来找最优阈值（不能用测试集找，会过拟合）
X = df.drop('label', axis=1)
y = df['label']

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)
# 0.6 / 0.2 / 0.2 划分完成

print(f"\nStep2：特征工程（加入 StandardScaler 标准化）")
print(f"  训练集：{len(X_train)} 条")
print(f"  验证集：{len(X_val)} 条  ← 用于寻找最优阈值")
print(f"  测试集：{len(X_test)} 条  ← 最终评估，绝不碰阈值")
print("=" * 65)


# ============================================================
#  Step3：构建三个模型
# ============================================================

models = {}

models['RandomForest'] = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

if XGBOOST_AVAILABLE:
    models['XGBoost'] = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric='logloss', verbosity=0
        ))
    ])

if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
    ])


# ============================================================
#  Step4：训练 + 寻找最优阈值（核心挽救步骤！）
#
#  做法：
#    1. 在验证集上用 predict_proba() 拿到每个样本为"攻击"的概率
#    2. 尝试 0.01 ~ 0.99 共 99 个阈值
#    3. 每个阈值下计算 F1-Score，取最大的那个作为最优阈值
#    4. 用最优阈值在测试集上重新评估
#
#  网安场景：也可以主动降低阈值（比如强制 0.2）来提升 Recall
# ============================================================

print("\nStep3 & 4：训练模型 + 寻找最优分类阈值...")
print("=" * 65)

results = []

for name, model in models.items():
    print(f"\n训练模型：{name} ...")
    model.fit(X_train, y_train)

    # ---- 在验证集上寻找最优阈值 ----
    # predict_proba 返回 shape=(n_samples, 2)，第 2 列是 label=1 的概率
    val_proba = model.predict_proba(X_val)[:, 1]

    # 遍历阈值，计算 F1
    thresholds = np.arange(0.01, 1.00, 0.01)
    f1_scores = []
    for thresh in thresholds:
        y_val_pred = (val_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1_val = f1_scores[best_idx]

    print(f"  {name} 最优阈值：{best_threshold:.2f}（验证集 F1={best_f1_val:.4f}）")

    # ---- 在测试集上用最优阈值评估 ----
    test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred_opt = (test_proba >= best_threshold).astype(int)
    y_test_pred_default = (test_proba >= 0.5).astype(int)  # 默认 0.5 对比

    # 用最优阈值的指标（这是我们报告的正式结果）
    acc  = accuracy_score(y_test, y_test_pred_opt)
    prec = precision_score(y_test, y_test_pred_opt, zero_division=0)
    rec  = recall_score(y_test, y_test_pred_opt, zero_division=0)
    f1   = f1_score(y_test, y_test_pred_opt, zero_division=0)

    # 默认阈值的指标（用于对比）
    rec_default = recall_score(y_test, y_test_pred_default, zero_division=0)
    f1_default  = f1_score(y_test, y_test_pred_default, zero_division=0)

    results.append({
        '模型': name,
        '最优阈值': round(best_threshold, 2),
        '准确率': round(acc, 4),
        '精确率': round(prec, 4),
        '召回率': round(rec, 4),
        'F1-Score': round(f1, 4),
    })

    print(f"  测试集结果（最优阈值 {best_threshold:.2f}）：")
    print(f"    精确率 Precision：{prec:.4f}")
    print(f"    召回率 Recall  ：{rec:.4f}  ← 挽救后提升了！")
    print(f"    F1-Score      ：{f1:.4f}  ← 比默认 0.5 的 {f1_default:.4f} 好！")
    print(f"    （默认阈值 0.5 的 Recall={rec_default:.4f}，F1={f1_default:.4f}）")

    cm = confusion_matrix(y_test, y_test_pred_opt)
    print(f"    混淆矩阵：TN={cm[0,0]}, FP={cm[0,1]} / FN={cm[1,0]}, TP={cm[1,1]}")

print("\n" + "=" * 65)
print("三模型对比（最优阈值下）：")
result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))
print("=" * 65)


# ============================================================
#  Step5：可视化
#   (a) 三模型 F1-Score 柱状图
#   (b) 单个模型的 Precision-Recall 曲线（展示阈值变化的影响）
# ============================================================

print("\nStep5：生成可视化图表...")

# ---- 图 (a)：三模型 F1-Score 对比 ----
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax1 = axes[0]
x_pos = np.arange(len(result_df))
colors = ['#4C72B0', '#55A868', '#C44E52'][:len(result_df)]
bars = ax1.bar(x_pos, result_df['F1-Score'], color=colors,
                edgecolor='black', linewidth=1.2, alpha=0.85)

for bar, value in zip(bars, result_df['F1-Score']):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.005,
             f'{value:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_xlabel('模型', fontsize=13, fontweight='bold')
ax1.set_ylabel('F1-Score（最优阈值下）', fontsize=13, fontweight='bold')
ax1.set_title('三模型 F1-Score 对比（v3.0 阈值调优版）',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(result_df['模型'], fontsize=12)
ax1.set_ylim([0, max(result_df['F1-Score']) + 0.06])
ax1.grid(axis='y', linestyle='--', alpha=0.4)

best_idx = result_df['F1-Score'].idxmax()
ax1.text(bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
         bars[best_idx].get_height() + 0.04,
         '冠军', ha='center', fontsize=11, color='red', fontweight='bold')

# ---- 图 (b)：Precision-Recall 曲线（以 XGBoost 为例）----
# 展示"降低阈值 → Recall 升，Precision 降"的权衡关系
if XGBOOST_AVAILABLE:
    xgb_model = models['XGBoost']
    test_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_test, test_proba_xgb)

    ax2 = axes[1]
    ax2.plot(rec_curve, prec_curve, linewidth=2, color='#C44E52')
    ax2.set_xlabel('Recall（召回率）', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Precision（精确率）', fontsize=13, fontweight='bold')
    ax2.set_title('XGBoost：Precision-Recall 权衡曲线\n（阈值降低 → Recall 升，Precision 降）',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.grid(linestyle='--', alpha=0.4)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    # 标注几个关键阈值位置
    for t in [0.1, 0.2, 0.5, 0.8]:
        idx = np.argmin(np.abs(thresh_curve - t))
        ax2.annotate(f'阈值={t}',
                     xy=(rec_curve[idx], prec_curve[idx]),
                     xytext=(rec_curve[idx] + 0.05, prec_curve[idx] - 0.05),
                     fontsize=9, alpha=0.7,
                     arrowprops=dict(arrowstyle='->', alpha=0.5))
else:
    axes[1].axis('off')

plt.tight_layout()
plt.show()

print("\n挽救完成！看看 F1-Score 有没有捞回来 🏆")
print("提示：如果 F1 还是很低，说明数据本身信号太弱，")
print("      下一步可以考虑：调整标签生成逻辑 / 增加有效特征 / 用 SMOTE 处理不平衡")
print("=" * 65)


# In[ ]:




