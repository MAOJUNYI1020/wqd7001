import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 读取数据
file_path = 'D:/project/wqd7001/Explore/aug_1_31_weather_traffic.csv'
df = pd.read_csv(file_path)

# 输出图路径
output_dir = 'D:/project/wqd7001/Explore'
os.makedirs(output_dir, exist_ok=True)

# 标签划分函数
def classify_rain(p):
    if p == 0:
        return 'No Rain'
    elif p <= 0.2:
        return 'Light Rain'
    elif p <= 0.6:
        return 'Moderate Rain'
    else:
        return 'Heavy Rain'

def classify_temp(t):
    if t < 28:
        return 'Cool'
    elif t < 33:
        return 'Warm'
    else:
        return 'Hot'

def classify_wind(w):
    if w < 1.5:
        return 'Low Wind'
    elif w < 3:
        return 'Moderate Wind'
    else:
        return 'Strong Wind'

def classify_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    else:
        return 'Poor'

# 应用标签
df['Rain Level'] = df['precip'].apply(classify_rain)
df['Temp Level'] = df['temp'].apply(classify_temp)
df['Wind Level'] = df['windspeed'].apply(classify_wind)
df['AQI Level'] = df['aqi'].apply(classify_aqi)

# 添加 weekday/weekend
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day_type'] = df['datetime'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

# 热力图函数
def draw_correlation_heatmap_with_rain_tags(street_name):
    subset = df[df['street'] == street_name]

    # 数值相关性
    corr = subset[['travelTimeRatio', 'precip', 'temp', 'windspeed', 'aqi']].corr()
    corr = corr[['travelTimeRatio']].sort_values(by='travelTimeRatio', ascending=False)

    # 降雨等级相关性
    rain_dummies = pd.get_dummies(subset['Rain Level'])
    rain_corr = rain_dummies.apply(lambda x: subset['travelTimeRatio'].corr(x))
    rain_corr.name = 'travelTimeRatio'
    rain_corr_df = pd.DataFrame(rain_corr)

    # 合并 & 排序
    final_corr = pd.concat([corr, rain_corr_df])
    final_corr = final_corr[~final_corr.index.duplicated(keep='first')]
    final_corr = final_corr.sort_values(by='travelTimeRatio', ascending=False)

    # 绘图
    plt.figure(figsize=(6, 6))
    sns.heatmap(final_corr, annot=True, cmap='RdBu_r', center=0, fmt=".2f", cbar=True,
                linewidths=0.5, linecolor='white')
    plt.title(f'Correlation Heatmap for {street_name}')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{street_name.replace(' ', '_')}_detailed_corr_heatmap.png")
    plt.close()

# 箱线图函数
def draw_boxplot(col, label):
    for street in df['street'].unique():
        subset = df[df['street'] == street]
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=subset, x='hour', y='travelTimeRatio', hue=col)
        plt.title(f'{label} vs Travel Time Ratio by Hour ({street})')
        plt.xlabel('Hour of Day')
        plt.ylabel('Travel Time Ratio')
        plt.legend(title=label)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{label.replace(" ", "_")}_hourly_boxplot_{street.replace(" ", "_")}.png')
        plt.close()

# weekday vs weekend
def draw_weekday_vs_weekend():
    for street in df['street'].unique():
        subset = df[df['street'] == street]
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=subset, x='hour', y='travelTimeRatio', hue='day_type')
        plt.title(f'Weekday vs Weekend Travel Time Ratio by Hour ({street})')
        plt.xlabel('Hour of Day')
        plt.ylabel('Travel Time Ratio')
        plt.legend(title='Day Type')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/weekday_weekend_hourly_boxplot_{street.replace(" ", "_")}.png')
        plt.close()

# 执行所有图生成
for street in df['street'].unique():
    draw_correlation_heatmap_with_rain_tags(street)

draw_boxplot('Rain Level', 'Rain Level')
draw_boxplot('Temp Level', 'Temperature Level')
draw_boxplot('Wind Level', 'Wind Level')
draw_boxplot('AQI Level', 'AQI Level')
draw_weekday_vs_weekend()
