import sys, pandas as pd

MAIN_DIR = sys.argv[1] #'data/run3_768x768'
files = ['log_1XSR3.csv', 'log_a3LNR.csv', 'log_iToyU.csv', 'log_N2ps3.csv', 
         'log_NXqy6.csv', 'log_8mLBw.csv', 'log_gfizK.csv', 'log_Jp5OJ.csv',
         'log_n3cwr.csv', 'log_UuDO9.csv']

logs = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{files[0]}')
for file in files[1:]:
    log = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{file}')
    logs = pd.concat([logs, log], ignore_index=True)

logs.to_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/img_log.csv', index=False)
