import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------- 改进版 MLP（小批量 + 动量 + 早停 + 加权L2）----------------------------
class MLPRegressor:
    def __init__(self, input_size, hidden_sizes, output_size=1, learning_rate=0.01,
                 epochs=1000, lambda_reg=0.0, momentum=0.9, batch_size=256,
                 patience=20, tol=1e-4):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.lr = learning_rate
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.momentum = momentum
        self.batch_size = batch_size
        self.patience = patience
        self.tol = tol

        # 构建网络结构
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(layer_sizes) - 1

        # He 初始化
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            w = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
            b = np.zeros((1, out_dim))
            self.weights.append(w)
            self.biases.append(b)

        # 动量缓存
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.v_biases = [np.zeros_like(b) for b in self.biases]

        # 早停相关
        self.best_weights = None
        self.best_biases = None
        self.best_loss = np.inf
        self.wait = 0

        self.loss_history = []       # 每个 epoch 的平均训练损失（加权+L2）
        self.val_loss_history = []   # 每个 epoch 的验证损失（原始 MSE，无正则化）

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, a):
        return (a > 0).astype(float)

    def forward(self, X):
        """X: (n_samples, input_size) 返回预测值 (n_samples, output_size)"""
        self.z = []
        self.a = [X]
        current = X
        for i in range(self.num_layers - 1):
            z_i = current @ self.weights[i] + self.biases[i]
            a_i = self.relu(z_i)
            self.z.append(z_i)
            self.a.append(a_i)
            current = a_i
        z_out = current @ self.weights[-1] + self.biases[-1]
        self.z.append(z_out)
        self.a.append(z_out)
        return z_out

    def compute_loss(self, y_true, y_pred, sample_weight=None, add_reg=True):
        """加权 MSE + 可选的 L2 正则化"""
        m = y_true.shape[0]
        diff = y_pred - y_true
        squared = diff ** 2
        if sample_weight is not None:
            weighted_squared = squared * sample_weight
            mse = np.mean(weighted_squared)
        else:
            mse = np.mean(squared)

        if add_reg and self.lambda_reg > 0:
            l2_reg = 0.0
            for w in self.weights:
                l2_reg += np.sum(w ** 2)
            l2_reg *= (self.lambda_reg / 2.0)
            return mse + l2_reg
        return mse

    def backward(self, X, y_true, y_pred, sample_weight=None):
        """计算梯度（带 L2 正则化贡献）"""
        m = X.shape[0]
        if sample_weight is not None:
            dL_dz2 = (2.0 / m) * (y_pred - y_true) * sample_weight
        else:
            dL_dz2 = (2.0 / m) * (y_pred - y_true)

        dW_list = [None] * self.num_layers
        db_list = [None] * self.num_layers

        for i in reversed(range(self.num_layers)):
            a_prev = self.a[i]   # (n, in_dim)
            if i == self.num_layers - 1:   # 输出层
                dW = a_prev.T @ dL_dz2
                db = np.sum(dL_dz2, axis=0, keepdims=True)
                dL_dz_prev = dL_dz2 @ self.weights[i].T
            else:                           # 隐藏层
                dL_da = dL_dz2
                dL_dz = dL_da * self.relu_derivative(self.z[i])
                dW = a_prev.T @ dL_dz
                db = np.sum(dL_dz, axis=0, keepdims=True)
                dL_dz_prev = dL_dz @ self.weights[i].T

            # L2 正则化梯度
            if self.lambda_reg > 0:
                dW += self.lambda_reg * self.weights[i]

            dW_list[i] = dW
            db_list[i] = db
            dL_dz2 = dL_dz_prev

        return dW_list, db_list

    def update_with_momentum(self, dW_list, db_list):
        """动量更新"""
        for i in range(self.num_layers):
            self.v_weights[i] = self.momentum * self.v_weights[i] - self.lr * dW_list[i]
            self.v_biases[i] = self.momentum * self.v_biases[i] - self.lr * db_list[i]
            self.weights[i] += self.v_weights[i]
            self.biases[i] += self.v_biases[i]

    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None, verbose=True):
        n_samples = X_train.shape[0]
        best_val_loss = np.inf
        wait = 0
        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.biases]

        for epoch in range(self.epochs):
            # 学习率衰减（每 1000 个 epoch 衰减 0.9）
            if epoch % 300 == 0 and epoch != 0:
                self.lr *= 0.9

            # ---------- 小批量训练 ----------
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            w_shuffled = sample_weight[indices] if sample_weight is not None else None

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                w_batch = w_shuffled[start:end] if sample_weight is not None else None

                y_pred_batch = self.forward(X_batch)
                loss_batch = self.compute_loss(y_batch, y_pred_batch, w_batch, add_reg=True)
                epoch_loss += loss_batch
                n_batches += 1

                dW, db = self.backward(X_batch, y_batch, y_pred_batch, w_batch)
                self.update_with_momentum(dW, db)

            avg_train_loss = epoch_loss / n_batches
            self.loss_history.append(avg_train_loss)

            # 验证集评估（无正则化，原始 MSE）
            y_pred_val = self.forward(X_val)
            val_loss = self.compute_loss(y_val, y_pred_val, add_reg=False)
            self.val_loss_history.append(val_loss)

            # 早停判断
            if val_loss < best_val_loss - self.tol:
                best_val_loss = val_loss
                wait = 0
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                wait += 1
                if wait >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}, best val loss = {best_val_loss:.6f}")
                    break

            if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch:4d}/{self.epochs}, Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 恢复最佳参数
        self.weights = best_weights
        self.biases = best_biases

    def predict(self, X):
        return self.forward(X)


# ---------------------------- 主程序 ----------------------------
# 1. 加载数据
df = pd.read_csv('MLP_data.csv')
print("数据前5行：")
print(df.head())

feature_cols = ['longitude', 'latitude', 'housing_age', 'homeowner_income']
X_raw = df[feature_cols].values
y_raw = df['house_price'].values.reshape(-1, 1)

# 对目标取对数
y_log = np.log1p(y_raw)

# 2. 划分训练集（60%）、验证集（20%）、测试集（20%）
X_temp, X_test, y_temp, y_test = train_test_split(X_raw, y_log, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# 3. 特征标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# 4. 目标标准化（对数空间）
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# 5. 计算训练样本权重（基于原始房价）
y_train_raw = np.expm1(y_train)
weights_raw = y_train_raw / np.mean(y_train_raw)
weights = np.sqrt(weights_raw)
weights = np.clip(weights, 1.0, 2.0)
print(f"样本权重范围: [{weights.min():.2f}, {weights.max():.2f}]")

# 6. 创建模型（三层：128->64->32，参数量约1.1万）
input_dim = X_train_scaled.shape[1]
hidden_sizes = [256, 128, 64]          # 三层
lr = 0.011
epochs = 3000
lambda_reg = 0.001                    # L2 系数
momentum = 0.9
batch_size = 256
patience = 50                         # 稍微提高耐心，让模型多探索
tol = 1e-6

model = MLPRegressor(input_size=input_dim, hidden_sizes=hidden_sizes,
                     output_size=1, learning_rate=lr, epochs=epochs,
                     lambda_reg=lambda_reg, momentum=momentum, batch_size=batch_size,
                     patience=patience, tol=tol)

print("\n开始训练 (小批量 + 动量SGD + 早停 + 加权MSE + L2)...")
model.fit(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
          sample_weight=weights, verbose=True)

# 7. 绘制损失曲线（从100轮开始，下降和收敛更明显）
plt.figure(figsize=(10, 5))
epochs_range = range(100, len(model.loss_history))
plt.plot(epochs_range, model.loss_history[100:], label='Training Loss (Weighted MSE+L2)', linewidth=2)
plt.plot(epochs_range, model.val_loss_history[100:], label='Validation Loss (raw MSE)', linestyle='--', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve (Mini-batch + Momentum + Early Stopping)')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('loss_curve_final.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 测试集预测与反标准化
y_pred_scaled = model.predict(X_test_scaled)
y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(scaler_y.inverse_transform(y_test_scaled))

y_pred_1d = y_pred.ravel()
y_test_1d = y_test_orig.ravel()

mse = mean_squared_error(y_test_1d, y_pred_1d)
mae = mean_absolute_error(y_test_1d, y_pred_1d)
r2 = r2_score(y_test_1d, y_pred_1d)
avg_price = np.mean(y_test_1d)

print("\n========== 测试集结果 ==========")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² : {r2:.4f}")
print(f"平均房价: {avg_price:.2f}")
print(f"MAE / 平均房价 = {mae / avg_price * 100:.2f}%")

# 9. 散点图
plt.figure(figsize=(7, 6))
plt.scatter(y_test_1d, y_pred_1d, alpha=0.5, s=10)
min_val = min(y_test_1d.min(), y_pred_1d.min())
max_val = max(y_test_1d.max(), y_pred_1d.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.title(f'True vs Predicted (R²={r2:.4f})')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_final.png', dpi=300, bbox_inches='tight')
plt.show()

print("图片已保存：loss_curve_final.png 和 scatter_final.png")