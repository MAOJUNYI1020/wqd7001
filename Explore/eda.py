import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


save_dir = Path("D:/project/wqd7001/Explore")
save_dir.mkdir(parents=True, exist_ok=True)


df = pd.read_csv("../Scrub/3months_weather_traffic_processed.csv")


df["street"] = df["street_Jalan Sultan Salahuddin"].apply(lambda x: "JSS" if x == 1 else "JLT")

# 恢复 travelTimeRatio（用于图示，不参与建模）
def recover_ttr(cl):
    if cl == 0:
        return np.random.uniform(0.85, 1.00)
    elif cl == 1:
        return np.random.uniform(1.01, 1.20)
    else:
        return np.random.uniform(1.21, 1.50)
df["travelTimeRatio"] = df["congestion_level"].apply(recover_ttr)

# 降雨分级
def categorize_precip(p):
    if p == 0:
        return "No Rain"
    elif p < 2:
        return "Light"
    elif p < 5:
        return "Moderate"
    else:
        return "Heavy"
df["precip_level"] = df["precip"].apply(categorize_precip)

# 设置风格
sns.set(style="whitegrid")

# 1. 单变量直方图
plt.figure(figsize=(16, 12))
num_cols = ['temp', 'humidity', 'windspeed', 'aqi', 'precip', 'hour', 'weekday']
for i, col in enumerate(num_cols):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(save_dir / "plot_1_histograms.png")
plt.close()

# 2. 每小时 travelTimeRatio 折线图
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="hour", y="travelTimeRatio", estimator="mean", ci=None)
plt.title("Average Travel Time Ratio by Hour")
plt.ylabel("Avg TTR")
plt.xlabel("Hour of Day")
plt.grid(True)
plt.tight_layout()
plt.savefig(save_dir / "plot_2_ttr_by_hour.png")
plt.close()

# 3. 散点图矩阵（pairplot）
sns.pairplot(df[["temp", "humidity", "windspeed", "aqi", "precip", "travelTimeRatio"]], diag_kind="kde")
plt.suptitle("Pairplot of Features vs Travel Time Ratio", y=1.02)
plt.savefig(save_dir / "plot_3_pairplot.png")
plt.close()

# 4. 热力图
corr = df[["temp", "humidity", "windspeed", "aqi", "precip", "hour", "weekday", "travelTimeRatio"]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(save_dir / "plot_4_correlation_heatmap.png")
plt.close()

# 5. 街道对降雨响应：箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="precip_level", y="travelTimeRatio", hue="street")
plt.title("TTR by Rainfall Level and Street")
plt.tight_layout()
plt.savefig(save_dir / "plot_5_street_rain_boxplot.png")
plt.close()

# 6. 高峰 vs 非高峰 降雨响应
df["peak"] = df["hour"].apply(lambda h: "Peak" if h in [7, 8, 17, 18] else "Off-Peak")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="precip_level", y="travelTimeRatio", hue="peak")
plt.title("TTR by Rainfall and Peak Hours")
plt.tight_layout()
plt.savefig(save_dir / "plot_6_peak_rain_boxplot.png")
plt.close()

# 7. 工作日 vs 周末 降雨响应
df["weektype"] = df["weekday"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="precip_level", y="travelTimeRatio", hue="weektype")
plt.title("TTR by Rainfall Level and Weekday/Weekend")
plt.tight_layout()
plt.savefig(save_dir / "plot_7_weektype_rain_boxplot.png")
plt.close()

print("✅ 所有 EDA 图表已保存至 Explore 文件夹。")
