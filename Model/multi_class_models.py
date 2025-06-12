import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 1. 读取数据
df = pd.read_csv("../Scrub/3months_weather_traffic_processed.csv")

# 2. 特征与标签
X = df.drop(columns=["congestion_level"])
y = df["congestion_level"]

# 3. 划分训练/测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 输出目录
output_path = Path("./")
output_path.mkdir(parents=True, exist_ok=True)

# ==========================
# Logistic Regression + Balanced
# ==========================
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("📊 Logistic Regression Report (Balanced):\n")
print(classification_report(y_test, y_pred_lr))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr), display_labels=["Smooth", "Congested", "Heavy"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression (Balanced)")
plt.savefig(output_path / "cm_logistic_regression.png")
plt.close()

# ==========================
# Random Forest + Balanced
# ==========================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("📊 Random Forest Report (Balanced):\n")
print(classification_report(y_test, y_pred_rf))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf), display_labels=["Smooth", "Congested", "Heavy"])
disp.plot(cmap="Greens")
plt.title("Confusion Matrix - Random Forest (Balanced)")
plt.savefig(output_path / "cm_random_forest.png")
plt.close()

# ==========================
# Improved Neural Network (MLP)
# ==========================
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    alpha=0.0005,  # L2 正则
    learning_rate_init=0.001,
    max_iter=800,
    random_state=42
)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)

print("📊 Neural Network (Improved MLP) Report:\n")
print(classification_report(y_test, y_pred_mlp))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_mlp), display_labels=["Smooth", "Congested", "Heavy"])
disp.plot(cmap="Purples")
plt.title("Confusion Matrix - Improved Neural Network")
plt.savefig(output_path / "cm_neural_network.png")
plt.close()

print("✅ 所有模型已训练完成，图像保存至 Model 文件夹。")
