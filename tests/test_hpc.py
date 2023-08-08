import sys       
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
from diffusers import DiffusionPipeline
import torch, tomesd, numpy as np, pandas as pd, time
from libs.pytorch_fid.src.fid.fid_score import calc_fid_given_lists

assert torch.cuda.is_available()
pipeline = DiffusionPipeline.from_pretrained("/gpfs/project/hebal100/ba-code/pipelines/SD-v1-5").to('cuda')
pipeline.enable_attention_slicing()
def dummy(images, **kwargs):
    return images, [False]
pipeline.safety_checker = dummy

all_prompts = pd.read_csv('data/prompts.csv')['colummn'].values
idcs = np.random.randint(0, len(all_prompts), 1)[0]
prompt = all_prompts[idcs]
seed = int(np.random.randint(0, 4294967295, 1)[0])

x, y = 832, 832
image_c1 = pipeline(prompt, x, y, generator=torch.Generator().manual_seed(seed)).images[0]
tomesd.apply_patch(pipeline, 0.5, merge_attn=True, merge_crossattn=False, merge_mlp=False)
image_m1 = pipeline(prompt, x, y, generator=torch.Generator().manual_seed(seed)).images[0]
tomesd.remove_patch(pipeline)

fid1 = calc_fid_given_lists(image_c1, image_m1)

print(fid1)

'''logs = [[prompt, seed, 0, t0]]
fid1 = calc_fid_given_lists(image_c1, image_m1)
logs.append([prompt, seed, fid1, t1])

log = pd.DataFrame(logs, columns=['prompt', 'seed', 'fid', 'time'])
log.to_csv(f'/gpfs/scratch/hebal100/data/test/test5/log.csv', index=False)
'''

