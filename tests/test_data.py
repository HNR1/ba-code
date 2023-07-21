import sys       
sys.path.insert('/gpfs/project/hebal100/ba-code/libs')
sys.path.insert('/gpfs/project/hebal100/ba-code')
from diffusers import DiffusionPipeline
from datasets import DatasetDict
import torch, numpy as np
from tomesd import tomesd
from pytorch_fid.src.fid.fid_score import calc_fid_given_lists

assert torch.cuda.is_available()
pipeline = DiffusionPipeline.from_pretrained("pipelines/SD-v1-5").to('cuda')
pipeline.enable_attention_slicing()

all_prompts = DatasetDict.from_parquet("data/prompts_hf.parquet")
num_prompts = 10
prompts = all_prompts[:num_prompts].get('Prompt')
seeds = np.random.randint(0, 4294967295, len(prompts))

num_imgs = 2
assert num_imgs <= num_prompts
images = []

def cut_prompt(prompt, length, char):
    idx = prompt[:length].rfind(char)
    return prompt[:idx]

x, y = 768, 768
for i in range(num_imgs):
    prompt = cut_prompt(prompts[i], 300, ',')
    print(prompt)
    image = pipeline(prompt, x, y, generator=torch.Generator().manual_seed(seeds[i].item())).images[0]
    images.append(image)

print(len(images))
