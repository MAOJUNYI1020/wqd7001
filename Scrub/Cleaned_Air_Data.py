import pandas as pd

# 读取空气质量原始数据
air_df = pd.read_csv("../Obtain/kl_aug_air_quality.csv")

print("原始数据预览：")
print(air_df.head())

#保留必要字段并转换类型
required_cols = ['datetime', 'aqi']
air_df = air_df[required_cols]
air_df['datetime'] = pd.to_datetime(air_df['datetime'])

#缺失值处理（使用均值填充）
print("\n缺失值汇总：")
print(air_df.isnull().sum())

# 用平均值填补 aqi 缺失
if air_df['aqi'].isnull().any():
    mean_aqi = air_df['aqi'].mean()
    air_df['aqi'].fillna(mean_aqi, inplace=True)
    print(f"\n已用均值 {mean_aqi:.2f} 填补 aqi 缺失值")

# 去除重复行
air_df.drop_duplicates(inplace=True)

# 时间排序
air_df.sort_values(by='datetime', inplace=True)

# 保存清洗后数据
air_df.to_csv("../Scrub/cleaned_kl_air.csv", index=False)

print("\n清洗后预览：")
print(air_df.head())
