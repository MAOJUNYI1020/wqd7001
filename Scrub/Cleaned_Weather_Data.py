import pandas as pd


weather_df = pd.read_csv("../Obtain/kl_aug_weather.csv")

columns_to_keep = [
    'datetime', 'temp', 'feelslike', 'humidity', 'precip',
    'precipprob', 'windspeed', 'cloudcover', 'visibility', 'uvindex'
]
weather_df = weather_df[columns_to_keep]

# 将 'datetime' 转为 pandas 的 datetime 类型
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

#检查是否有缺失值
missing_summary = weather_df.isnull().sum()
print("缺失值汇总：\n", missing_summary)

# 删除含缺失值的行（如果有）
weather_df = weather_df.dropna()

print("清洗后的天气数据预览：\n", weather_df.head())

weather_df.to_csv("../Scrub/cleaned_kl_weather.csv", index=False)
print("已保存为 cleaned_kl_weather.csv")
