import sys       
import pandas as pd

PATH = sys.argv[1] #'data/run2_768x768'
directories = ['images_0', 'images_10', 'images_20', 'images_30', 'images_40', 'images_50', 'images_60']
m_vols = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
time_values = []

for v in m_vols:
    df = pd.read_csv(f'/gpfs/scratch/hebal100/{PATH}/img_log.csv')
    df = df[df.m_vol == v]
    t = df['time'].values.mean()
    time_values.append([v, t])

time_log = pd.DataFrame(time_values, columns=['m_vol', 'time'])
time_log.to_csv(f'{PATH}/time_log.csv', index=False)