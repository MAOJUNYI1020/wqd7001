import pandas as pd


weather_df = pd.read_csv("../Obtain/kl_aug_weather.csv")

columns_to_keep = [
    'datetime', 'temp', 'feelslike', 'humidity', 'precip',
    'precipprob', 'windspeed', 'cloudcover', 'visibility', 'uvindex'
]
weather_df = weather_df[columns_to_keep]

weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

missing_summary = weather_df.isnull().sum()

weather_df = weather_df.dropna()

weather_df.to_csv("../Scrub/cleaned_kl_weather.csv", index=False)
