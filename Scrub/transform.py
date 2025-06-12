import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path

# 路径配置
input_path = Path("3months_weather_traffic.csv")
output_path = Path("3months_weather_traffic_processed.csv")

# Step 1: 读取数据
df = pd.read_csv(input_path, parse_dates=["datetime"])

# Step 2: 时间特征提取
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.weekday
df["is_weekend"] = df["weekday"] >= 5

# Step 3: 拥堵等级生成（分类标签）
def classify_congestion(ttr):
    if ttr <= 1.0:
        return 0  # 顺畅
    elif ttr <= 1.2:
        return 1  # 堵塞
    else:
        return 2  # 严重堵塞

df["congestion_level"] = df["travelTimeRatio"].apply(classify_congestion)

# Step 4: 特征选择
num_features = ["temp", "humidity", "windspeed", "aqi", "precip"]
cat_features = ["street"]
time_features = ["hour", "weekday", "is_weekend"]

# Step 5: 标准化 + 编码
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
], remainder="passthrough")  # 保留时间字段

X = df[num_features + cat_features + time_features]
X_transformed = preprocessor.fit_transform(X)

# Step 6: 构造输出 DataFrame
encoded_cat = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)
final_columns = num_features + list(encoded_cat) + time_features

df_processed = pd.DataFrame(X_transformed, columns=final_columns)
df_processed["congestion_level"] = df["congestion_level"]

# Step 7: 保存处理后的文件
df_processed.to_csv(output_path, index=False)
print(f"✅ 成功生成: {output_path}")
