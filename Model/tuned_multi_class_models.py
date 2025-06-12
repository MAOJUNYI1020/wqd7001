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

# 4. 图像保存路径
output_path = Path(".")
output_path.mkdir(parents=True, exist_ok=True)

# ================================
# Logistic Regression (调参版)
# ================================
lr_model = LogisticRegression(C=0.5, class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Report (Tuned):\n")
print(classification_report(y_test, y_pred_lr))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lr), display_labels=["Smooth", "Congested", "Heavy"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression (Tuned)")
plt.savefig(output_path / "cm_logistic_regression_tuned.png")
plt.close()

# ================================
# Random Forest (调参版)
# ================================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Report (Tuned):\n")
print(classification_report(y_test, y_pred_rf))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf), display_labels=["Smooth", "Congested", "Heavy"])
disp.plot(cmap="Greens")
plt.title("Confusion Matrix - Random Forest (Tuned)")
plt.savefig(output_path / "cm_random_forest_tuned.png")
plt.close()

# ================================
# Neural Network (调参版)
# ================================
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)

print("Neural Network (MLP) Report (Tuned):\n")
print(classification_report(y_test, y_pred_mlp))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_mlp), display_labels=["Smooth", "Congested", "Heavy"])
disp.plot(cmap="Purples")
plt.title("Confusion Matrix - Neural Network (Tuned)")
plt.savefig(output_path / "cm_neural_network_tuned.png")
plt.close()

print("所有调参后的模型已训练完成，混淆矩阵图保存至 ./Model 文件夹。")
