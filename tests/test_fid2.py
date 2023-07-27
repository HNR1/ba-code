import sys       
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
from libs.pytorch_fid.src.fid.fid_score import calculate_fid_given_paths

PATH = '/gpfs/scratch/hebal100/data/run2_768x768'
directories = ['images_0', 'images_10', 'images_20', 'images_30', 'images_40', 'images_50', 'images_60']
fid_values = []

fid = calculate_fid_given_paths((f'{PATH}/images_0', f'{PATH}/images_10'), batch_size=50, device='cuda', dims=2048)

print(fid)