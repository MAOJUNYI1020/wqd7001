import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 输入 / 输出路径
input_path = "D:/project/wqd7001/Scrub/aug_1_31_weather_traffic.csv"
output_path = "D:/project/wqd7001/Scrub/processed_aug_weather_traffic.csv"

# 读取原始数据
df = pd.read_csv(input_path)

# 时间特征处理
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df = df.drop(columns=['datetime'])

# 标准化列
standardize_cols = [
    'temp', 'feelslike', 'windspeed', 'visibility', 'aqi',
    'averageSpeed', 'medianSpeed', 'harmonicAverageSpeed',
    'averageTravelTime'
]

# 归一化列
normalize_cols = [
    'humidity', 'precip', 'precipprob', 'cloudcover', 'uvindex', 'sampleSize'
]

# 应用标准化与归一化
scaler_std = StandardScaler()
df[standardize_cols] = scaler_std.fit_transform(df[standardize_cols])

scaler_minmax = MinMaxScaler()
df[normalize_cols] = scaler_minmax.fit_transform(df[normalize_cols])

# One-Hot 编码（只对 street 和 weekday）
categorical_cols = ['street', 'weekday']
df = pd.get_dummies(df, columns=categorical_cols)

# ✅ 将所有 bool 列转成 int（True/False → 1/0）
df = df.astype({col: int for col in df.columns if df[col].dtype == 'bool'})

# 保存处理后数据
df.to_csv(output_path, index=False)
print("✅ 数据预处理完成，文件已保存至：", output_path)
