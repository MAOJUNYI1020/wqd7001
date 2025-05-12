import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math

# ======= 文件路径配置 =======
csv_path = "D:/project/wqd7001/Scrub/processed_aug_weather_traffic.csv"
output_dir = "D:/project/wqd7001/Explore/"
os.makedirs(output_dir, exist_ok=True)

# ======= 读取数据 =======
df = pd.read_csv(csv_path)
df['hour'] = df['hour'].round().astype(int)
df['weekday'] = df[[c for c in df.columns if c.startswith("weekday_")]].idxmax(axis=1).str.replace("weekday_", "")
df['street'] = df[[c for c in df.columns if c.startswith("street_")]].idxmax(axis=1).str.replace("street_", "")
df['is_weekend'] = df['is_weekend'].astype(int)
df['DayType'] = df['is_weekend'].map({1: "Weekend", 0: "Weekday"})

# ======= 降雨等级划分函数 =======
def assign_rain_level(data):
    max_precip = data['precip'].max()
    def label(p):
        if p == 0:
            return 'No Rain'
        elif p <= max_precip / 3:
            return 'Light Rain'
        elif p <= 2 * max_precip / 3:
            return 'Moderate Rain'
        else:
            return 'Heavy Rain'
    return data['precip'].apply(label)

# ======= 每条街道图组生成（无 Pairplot）=======
def generate_all_plots_for_street(data):
    plots = []

    # 1. Histogram + KDE
    fig = plt.figure()
    sns.histplot(data['travelTimeRatio'], kde=True)
    plt.title("Distribution of Travel Time Ratio")
    plots.append(fig)

    # 2. Boxplot
    fig = plt.figure()
    sns.boxplot(y=data['travelTimeRatio'])
    plt.title("Boxplot of Travel Time Ratio")
    plots.append(fig)

    # 3. Correlation Heatmap
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(data.select_dtypes(include='number').corr(), cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap")
    plots.append(fig)

    # 4. Temp vs TravelTimeRatio
    fig = plt.figure()
    sns.scatterplot(x='temp', y='travelTimeRatio', data=data)
    plt.title("Temperature vs Travel Time Ratio")
    plots.append(fig)

    # 5. Weekday vs TravelTimeRatio
    fig = plt.figure()
    sns.boxplot(x='weekday', y='travelTimeRatio', data=data)
    plt.title("Travel Time Ratio by Weekday")
    plots.append(fig)

    # 6. Rain Level vs TravelTimeRatio
    df_rain = data.copy()
    df_rain['Rain_Level'] = assign_rain_level(df_rain)
    fig = plt.figure()
    sns.boxplot(x='Rain_Level', y='travelTimeRatio', data=df_rain)
    plt.title("Travel Time Ratio by Rain Level")
    plots.append(fig)

    # 7. Hourly Avg
    fig = plt.figure()
    sns.lineplot(x='hour', y='travelTimeRatio', data=data, estimator='mean', errorbar=None, marker='o')
    plt.title("Avg Travel Time Ratio by Hour")
    plots.append(fig)

    # 8. Precipitation Bin Lineplot
    df_p = data.copy()
    df_p['precip_bin'] = pd.cut(df_p['precip'], bins=[0, 0.1, 0.3, 0.6, 1.0], labels=['Light', 'Moderate', 'Heavy', 'Extreme'])
    df_avg = df_p.groupby('precip_bin', observed=False)['travelTimeRatio'].mean().reset_index()
    fig = plt.figure()
    sns.lineplot(x='precip_bin', y='travelTimeRatio', data=df_avg, marker='o')
    plt.title("Avg Travel Time Ratio by Precipitation Level")
    plots.append(fig)

    # 9. DayType × Hour
    fig = plt.figure(figsize=(10, 5))
    sns.boxplot(x='hour', y='travelTimeRatio', hue='DayType', data=data)
    plt.title("Day Type vs Hour")
    plt.legend()
    plots.append(fig)

    # 10. Rain_Level × Hour
    df_hr = data.copy()
    df_hr['Rain_Level'] = assign_rain_level(df_hr)
    fig = plt.figure(figsize=(10, 5))
    sns.boxplot(x='hour', y='travelTimeRatio', hue='Rain_Level', data=df_hr)
    plt.title("Rain Level vs Hour")
    plt.legend()
    plots.append(fig)

    # 11. Hourly Trend
    fig = plt.figure()
    sns.lineplot(x='hour', y='travelTimeRatio', data=data, estimator='mean', errorbar=None)
    plt.title("Hourly Trend of Travel Time Ratio")
    plots.append(fig)

    return plots

# ======= 保存合并图组 =======
def save_combined_canvas(plots, out_file, cols=4):
    rows = math.ceil(len(plots) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()

    for i, p in enumerate(plots):
        p.canvas.draw()
        buf = p.canvas.buffer_rgba()
        img = np.asarray(buf)
        axes[i].imshow(img)
        axes[i].axis('off')
        plt.close(p)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✅ 合并图保存成功：{out_file}")

# ======= 单独生成 Pairplot 图 =======
def generate_pairplot(df, output_path):
    features = ['travelTimeRatio', 'temp', 'humidity', 'precip', 'windspeed']
    df_pair = df[features].dropna()
    sns.pairplot(df_pair)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Pairplot 图已保存：{output_path}")

# ======= 主执行逻辑 =======
generate_pairplot(df, os.path.join(output_dir, "pairplot_features.png"))

for street in df['street'].unique():
    subset = df[df['street'] == street]
    plots = generate_all_plots_for_street(subset)
    filename = f"all_plots_{street.replace(' ', '_')}.png"
    full_path = os.path.join(output_dir, filename)
    save_combined_canvas(plots, full_path, cols=4)
