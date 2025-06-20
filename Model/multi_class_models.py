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

# 2. 分离 X 与 y
X = df.drop(columns=["congestion_level"])
y = df["congestion_level"]

# 3. 划分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 输出图像保存路径：Model 文件夹
output_path = Path("./")
output_path.mkdir(parents=True, exist_ok=True)

# ==========================
# Logistic Regression
# ==========================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Report:\n")
print(classification_report(y_test, y_pred_lr))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr), display_labels=["Smooth", "Congested", "Heavy"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig(output_path / "cm_logistic_regression.png")
plt.close()

# ==========================
# Random Forest
# ==========================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Report:\n")
print(classification_report(y_test, y_pred_rf))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf), display_labels=["Smooth", "Congested", "Heavy"])
disp.plot(cmap="Greens")
plt.title("Confusion Matrix - Random Forest")
plt.savefig(output_path / "cm_random_forest.png")
plt.close()

# ==========================
# Neural Network (MLP)
# ==========================
mlp_model = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)

print("Neural Network (MLP) Report:\n")
print(classification_report(y_test, y_pred_mlp))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_mlp), display_labels=["Smooth", "Congested", "Heavy"])
disp.plot(cmap="Purples")
plt.title("Confusion Matrix - Neural Network (MLP)")
plt.savefig(output_path / "cm_neural_network.png")
plt.close()

print("所有模型训练完毕，混淆矩阵图已保存至 Model 文件夹。")
