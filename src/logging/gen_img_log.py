import pandas as pd

MAIN_DIR = 'data/run4'
files = ['log_6qrnm.csv', 'log_GhKAc.csv', 'log_HA0qr.csv', 'log_np6I8.csv', 
         'log_uC7tx.csv', 'log_dxoHq.csv', 'log_gr9JS.csv',  
         'log_lCvWs.csv', 'log_sOMGN.csv', 'log_zNTjN.csv']

logs = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{files[0]}')
for file in files[1:]:
    log = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{file}')
    logs = pd.concat([logs, log], ignore_index=True)

logs.to_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/img_log.csv', index=False)
