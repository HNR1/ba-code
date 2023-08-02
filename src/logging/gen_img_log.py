import pandas as pd

MAIN_DIR = 'data/run6'
files = ['log_0OI2J.csv', 'log_b8RLi.csv', 'log_F8hN7.csv', 'log_h0xPt.csv', 
         'log_QBA0b.csv', 'log_8rN8I.csv', 'log_bXR2e.csv', 'log_GOU8i.csv', 
         'log_pjxuw.csv', 'log_R9hB0.csv']

logs = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{files[0]}')
for file in files[1:]:
    log = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{file}')
    logs = pd.concat([logs, log], ignore_index=True)

logs.to_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/img_log.csv', index=False)
