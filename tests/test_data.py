import sys       
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
from diffusers import DiffusionPipeline
import torch, time, numpy as np
import pandas as pd
from tomesd import tomesd
# PATH = "/gpfs/scratch/hebal100/test"

assert torch.cuda.is_available()
pipeline = DiffusionPipeline.from_pretrained("pipelines/SD-v1-5").to('cuda')
pipeline.enable_attention_slicing()

all_prompts = pd.read_csv('data/prompts.csv')['colummn'].values
sample_size = 10
idcs = np.random.randint(0, len(all_prompts), sample_size)
prompts = all_prompts[idcs]
print(prompts[0])
seeds = np.random.randint(0, 4294967295, len(prompts))

num_imgs = sample_size
images = []

def cut_prompt(prompt, max_len=300, char=','):
    if len(prompt) <= max_len:
        return prompt
    idx = prompt[:max_len].rfind(char)
    # catch if char isn't used
    if idx <= 0:
        idx = prompt[:max_len].rfind(' ')
    return prompt[:idx]

x, y = 768, 768
# TODO: measure diffusion time
def gen_loop(prompts, x, y, seeds, num_imgs):    
    for i in range(num_imgs):
        prompt = cut_prompt(prompts[i])
        print(prompt)
        image = pipeline(prompt, x, y, generator=torch.Generator().manual_seed(seeds[i].item())).images[0]
        images.append(image)

gen_loop(prompts, x, y, seeds, num_imgs)
tomesd.apply_patch(pipeline, 0.1)
gen_loop(prompts, x, y, seeds, num_imgs)
tomesd.apply_patch(pipeline, 0.2)
gen_loop(prompts, x, y, seeds, num_imgs)
tomesd.apply_patch(pipeline, 0.3)
gen_loop(prompts, x, y, seeds, num_imgs)
tomesd.apply_patch(pipeline, 0.4)
gen_loop(prompts, x, y, seeds, num_imgs)
tomesd.apply_patch(pipeline, 0.5)
gen_loop(prompts, x, y, seeds, num_imgs)
tomesd.apply_patch(pipeline, 0.6)
gen_loop(prompts, x, y, seeds, num_imgs)
tomesd.remove_patch(pipeline)

print(len(images))
