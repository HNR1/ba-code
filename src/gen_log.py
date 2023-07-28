import sys, pandas as pd

MAIN_DIR = sys.argv[1] #'data/run2_768x768'
files = ['log_3HFsG.csv',  'log_jVDzd.csv',  'log_XAZgz.csv',
'log_48Nxf.csv',  'log_Qs4eF.csv',  'log_YQff8.csv',
'log_FJNhj.csv',  'log_rJPTw.csv',
'log_Jm4tN.csv',  'log_uboff.csv']

logs = pd.read_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{files[0]}')
for file in files[1:]:
    logs = pd.concat([logs, file], ignore_index=True)

log = pd.DataFrame(logs, columns=['prompt', 'seed', 'm_vol', 'time', 'name'])
log.to_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/img_log.csv', index=False)
