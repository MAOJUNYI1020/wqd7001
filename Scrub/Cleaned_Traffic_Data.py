import os
import json
import pandas as pd
from datetime import datetime, timedelta

# 设置文件夹路径
traffic_folder = r'D:\project\wqd7001\Obtain\kl_aug_traffic'
output_file = r'D:\project\wqd7001\Scrub\cleaned_kl_traffic.csv'

all_records = []

# 遍历所有 JSON 文件
for filename in sorted(os.listdir(traffic_folder), key=lambda x: int(x.split('.')[0])):
    if not filename.endswith('.json'):
        continue

    file_path = os.path.join(traffic_folder, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取对应的日期（1.json => 2024-08-01）
    day = int(filename.split('.')[0])
    base_date = datetime.strptime(f'2024-08-{day:02d}', '%Y-%m-%d')

    for segment in data['network']['segmentResults']:
        street = segment.get('streetName', 'Unknown')
        for t in segment.get('segmentTimeResults', []):
            time_set = t['timeSet']
            hour = time_set - 1  # 转换为小时：timeSet 2 => 1:00

            # 构造 datetime 字段
            dt = base_date + timedelta(hours=hour)

            record = {
                'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'street': street,
                'segmentId': segment['segmentId'],
                'averageSpeed': t.get('averageSpeed'),
                'medianSpeed': t.get('medianSpeed'),
                'harmonicAverageSpeed': t.get('harmonicAverageSpeed'),
                'travelTimeRatio': t.get('travelTimeRatio'),
                'averageTravelTime': t.get('averageTravelTime'),
                'sampleSize': t.get('sampleSize')
            }
            all_records.append(record)

# 转为 DataFrame
df = pd.DataFrame(all_records)

# 保存为 CSV 文件
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"✅ 已完成清洗并保存为：{output_file}")
