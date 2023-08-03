import sys       
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
from diffusers import DiffusionPipeline
import torch, tomesd, numpy as np, pandas as pd, time
from libs.pytorch_fid.src.fid.fid_score import calc_fid_given_lists

assert torch.cuda.is_available()
pipeline = DiffusionPipeline.from_pretrained("/gpfs/project/hebal100/ba-code/pipelines/SD-v1-5").to('cuda')
pipeline.enable_attention_slicing()

all_prompts = pd.read_csv('data/prompts.csv')['colummn'].values
idcs = np.random.randint(0, len(all_prompts), 1)[0]
prompt = all_prompts[idcs]
seed = np.random.randint(0, 4294967295, 1)[0]

x, y = 768, 768
start0 = time.time()
image_c1 = pipeline(prompt, x, y, generator=torch.Generator().manual_seed(seed)).images[0]
end0 = time.time()
t0 = end0 - start0
tomesd.apply_patch(pipeline, 0.5, merge_attn=True, merge_crossattn=False, merge_mlp=False)
start1 = time.time()
image_m1 = pipeline(prompt, x, y, generator=torch.Generator().manual_seed(seed)).images[0]
end1 = time.time()
t1 = end1 - start1
tomesd.apply_patch(pipeline, 0.5, merge_attn=False, merge_crossattn=True, merge_mlp=True)
start2 = time.time()
image_m2 = pipeline(prompt, x, y, generator=torch.Generator().manual_seed(seed)).images[0]
end2 = time.time()
t2 = end2 - start2
tomesd.remove_patch(pipeline)

logs = [[prompt, seed, 0, t0]]
fid1 = calc_fid_given_lists(image_c1, image_m1)
logs.append([prompt, seed, fid1, t1])
fid2 = calc_fid_given_lists(image_c1, image_m2)
logs.append([prompt, seed, fid2, t2])

log = pd.DataFrame(logs, columns=['prompt', 'seed', 'fid', 'time'])
log.to_csv(f'/gpfs/scratch/hebal100/data/test/test5/log.csv', index=False)


