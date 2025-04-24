import pandas as pd

# 读取清洗好的数据
traffic_df = pd.read_csv("D:/project/wqd7001/Scrub/cleaned_kl_traffic.csv", parse_dates=["datetime"])
air_df = pd.read_csv("D:/project/wqd7001/Scrub/cleaned_kl_air.csv", parse_dates=["datetime"])
weather_df = pd.read_csv("D:/project/wqd7001/Scrub/cleaned_kl_weather.csv", parse_dates=["datetime"])

# 先合并天气和空气质量（按时间对齐）
env_df = pd.merge(weather_df, air_df, on="datetime", how="inner")

# 再与交通数据合并
final_df = pd.merge(traffic_df, env_df, on="datetime", how="inner")

# 保存结果
final_df.to_csv("D:/project/wqd7001/Scrub/final_kl_dataset_1_17.csv", index=False)

print("合并完成")
