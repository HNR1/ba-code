import sys       
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
from diffusers import DiffusionPipeline
import torch, time, random, string, tomesd, numpy as np, pandas as pd

assert torch.cuda.is_available()
pipeline = DiffusionPipeline.from_pretrained("pipelines/SD-v1-5").to('cuda')
pipeline.enable_attention_slicing()
# disable safety checker
def dummy(images, **kwargs):
    return images, [False]
pipeline.safety_checker = dummy

all_prompts = pd.read_csv('data/prompts.csv')['colummn'].values
sample_size = 2
idcs = np.random.randint(0, len(all_prompts), sample_size)
prompts = all_prompts[idcs]
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
def gen_loop(pipeline, prompts, seeds, x, y, m_vol, num_imgs, dir, logger):    
    tomesd.apply_patch(pipeline, m_vol)
    for i in range(num_imgs):
        prompt, seed = cut_prompt(prompts[i]), seeds[i].item()
        start = time.time()
        image = pipeline(prompt, x, y, generator=torch.Generator().manual_seed(seed)).images[0]
        end = time.time()
        diff_time = end - start
        name = 'img_' + ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        #image.save(f"/gpfs/scratch/hebal100/test/test1/{dir}/{name}.png")
        logger.append([prompt, seed, m_vol, diff_time, name])

merge_volumes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
directories = ['images_0', 'images_10', 'images_20', 'images_30', 'images_40', 'images_50', 'images_60']
logger = []

for m_vol, dir in zip(merge_volumes, directories):
    gen_loop(pipeline, prompts, seeds, x, y, m_vol, num_imgs, dir, logger)
tomesd.remove_patch(pipeline)

log = pd.DataFrame(logger, columns=['prompt', 'seed', 'm_vol', 'time', 'name'])
name = 'log_' + ''.join(random.choices(string.ascii_letters + string.digits, k=5))
log.to_csv(f'/gpfs/scratch/hebal100/test/test1/logger/{name}.csv', index=False)
