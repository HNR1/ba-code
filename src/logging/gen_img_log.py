import sys
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
import pandas as pd

MAIN_DIR = sys.argv[1] #'data/run2'

files = ['log_0ez3h.csv', 'log_63ayC.csv', 'log_e34MW.csv', 'log_OlZmF.csv',
         'log_RUhNm.csv', 'log_4oO5e.csv', 'log_DcKZq.csv', 'log_Nsh7P.csv',
         'log_RNgEa.csv', 'log_VV081.csv']

logs = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{files[0]}')
for file in files[1:]:
    log = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{file}')
    logs = pd.concat([logs, log], ignore_index=True)

logs.to_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/img_log.csv', index=False)
