import sys       
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
import pandas as pd
from libs.pytorch_fid.src.fid.fid_score import calculate_fid_given_paths

PATH = sys.argv[1] #'data/run2_768x768'
directories = ['images_0', 'images_10', 'images_20', 'images_30', 'images_40', 'images_50', 'images_60']
fid_values = []

for dir in directories:
    fid = calculate_fid_given_paths((f'/gpfs/scratch/hebal100/{PATH}/images_0', 
                                     f'/gpfs/scratch/hebal100/{PATH}/{dir}'), batch_size=50, device='cuda', dims=2048)
    fid_values.append(fid)

fid_log = pd.DataFrame(fid_values, columns=['fid'])
fid_log.to_csv(f'/gpfs/scratch/hebal100/{PATH}/fid_log.csv', index=False)

