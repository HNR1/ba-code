import sys
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
import pandas as pd

MAIN_DIR = sys.argv[1] #'data/run2'

files = ['log_5WFyd.csv', 'log_Ddp27.csv', 'log_KzFEU.csv', 'log_xopa4.csv',
         'log_Z01zo.csv', 'log_9pZi6.csv', 'log_DZZnz.csv', 'log_mYH4x.csv',
         'log_xYRiz.csv',  'log_ZIt2j.csv']

logs = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{files[0]}')
for file in files[1:]:
    log = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{file}')
    logs = pd.concat([logs, log], ignore_index=True)

logs.to_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/img_log.csv', index=False)
