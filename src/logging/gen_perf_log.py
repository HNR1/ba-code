import sys       
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
import pandas as pd
from libs.pytorch_fid.src.fid.fid_score import calculate_fid_given_paths

PATH = sys.argv[1] #'data/run2_768x768'
directories = ['images_0', 'images_10', 'images_20', 'images_30', 'images_40', 'images_50', 'images_60']
m_vols = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
perf_values = []

for v, dir in zip(m_vols, directories):
    fid = calculate_fid_given_paths((f'/gpfs/scratch/hebal100/{PATH}/images_0', 
                                     f'/gpfs/scratch/hebal100/{PATH}/{dir}'), batch_size=50, device='cuda', dims=2048)
    df = pd.read_csv(f'/gpfs/scratch/hebal100/{PATH}/img_log.csv')
    df = df[df.m_vol == v]
    avg_time = df['time'].values.mean()
    perf_values.append([v, fid, avg_time])

perf_log = pd.DataFrame(perf_values, columns=['m_vol', 'fid', 'time'])
perf_log.to_csv(f'/gpfs/scratch/hebal100/{PATH}/performance_log.csv', index=False)
