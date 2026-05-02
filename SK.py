import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------- 1. 加载数据 ----------------------------
df = pd.read_csv('MLP_data.csv')
print("数据前5行：")
print(df.head())

feature_cols = ['longitude', 'latitude', 'housing_age', 'homeowner_income']
X = df[feature_cols].values
y = df['house_price'].values.reshape(-1, 1)

# ---------------------------- 2. 划分训练集与测试集 ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------- 3. 特征标准化 ----------------------------
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 也可对目标变量标准化（MLPRegressor内部并未要求，但标准化后训练更稳定）
# 先取对数
y_train_log = np.log1p(y_train)   # 或 np.log(y_train) 如果无0值
y_test_log = np.log1p(y_test)

# 2. 对对数房价标准化
scaler_y = StandardScaler()
y_train_log_scaled = scaler_y.fit_transform(y_train_log.reshape(-1, 1)).ravel()
y_test_log_scaled = scaler_y.transform(y_test_log.reshape(-1, 1)).ravel()

# ---------------------------- 4. 构建 MLP 模型（使用 Adam，容量充足）----------------------------
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),   # 三个隐藏层，神经元数充足
    activation='relu',                   # 非线性激活
    solver='adam',                       # Adam 优化器
    alpha=0.001,                         # L2 正则化系数
    learning_rate='adaptive',            # 验证损失停止下降时自动降低学习率
    learning_rate_init=0.001,            # 初始学习率
    max_iter=1000,                       # 最大迭代次数
    early_stopping=True,                 # 使用验证集早停
    validation_fraction=0.1,             # 从训练集中取 10% 作为验证集
    n_iter_no_change=20,                 # 连续 20 轮验证损失未改善则停止
    tol=1e-4,                            # 优化容忍度
    batch_size='auto',                   # 自动批大小（通常 200）
    random_state=42,
    verbose=True                         # 打印训练进度
)

print("\n开始训练 (Adam, 三层网络: 256→128→64)...")
mlp.fit(X_train_scaled, y_train_log_scaled)

# ---------------------------- 5. 损失曲线（sklearn 自动记录了 loss_curve_）----------------------------
plt.figure(figsize=(10, 5))
plt.plot(mlp.loss_curve_, label='Training Loss (MSE)', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Curve (MLPRegressor with Adam)')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('loss_curve_sklearn.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------- 6. 测试集预测与反标准化 ----------------------------
# 4. 预测
y_pred_log_scaled = mlp.predict(X_test_scaled)   # 标准化后的对数房价

# 5. 反标准化 -> 对数房价
y_pred_log = scaler_y.inverse_transform(y_pred_log_scaled.reshape(-1, 1)).ravel()

# 6. 指数还原 -> 原始房价
y_pred = np.expm1(y_pred_log)

# 真实值（用于评估）
y_test_orig = np.expm1(y_test_log)   # 或者直接用原始 y_test

# ---------------------------- 7. 评估指标 ----------------------------
mse = mean_squared_error(y_test_orig, y_pred)
mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)
avg_price = np.mean(y_test_orig)

print("\n========== 测试集结果 ==========")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² : {r2:.4f}")
print(f"平均房价: {avg_price:.2f}")
print(f"MAE / 平均房价 = {mae / avg_price * 100:.2f}%")

# ---------------------------- 8. 可视化：真实 vs 预测 + 残差分布 ----------------------------
# ---------------------------- 图1：预测值 vs 真实值散点图 ----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test_orig, y_pred, alpha=0.5, s=10)
min_val = min(y_test_orig.min(), y_pred.min())
max_val = max(y_test_orig.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.title(f'True vs Predicted (R² = {r2:.4f})')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_true_vs_pred_sk.png', dpi=300, bbox_inches='tight')
plt.show()   # 释放内存

print("图片已保存：scatter_true_vs_pred_sk.png")