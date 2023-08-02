import sys       
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
from diffusers import DiffusionPipeline
import torch, time, random, string, tomesd, numpy as np, pandas as pd

# save command line args
assert len(sys.argv) >= 5
MAIN_DIR = sys.argv[1]                      #'data/run5'
sample_size = int(sys.argv[2])              # 50
x, y =  int(sys.argv[3]), int(sys.argv[4])  # 768, 768
src_file = sys.argv[5]                      #'run2/logger/log_a5f7c.csv'

# load prompts and generate seeds
if src_file == None:
    all_prompts = pd.read_csv('data/prompts.csv')['colummn'].values
    idcs = np.random.randint(0, len(all_prompts), sample_size)
    prompts = all_prompts[idcs]
    seeds = np.random.randint(0, 4294967295, len(prompts))
# load prompts and seeds from previous run
else:
    prompts = pd.read_csv(f'/gpfs/scratch/hebal100/data/{src_file}')['prompt'].values
    seeds = pd.read_csv(f'/gpfs/scratch/hebal100/data/{src_file}')['seed'].values

# method for cutting oversized prompts
def cut_prompt(prompt, max_len=300, char=','):
    if len(prompt) <= max_len:
        return prompt
    
    idx = prompt[:max_len].rfind(char)
    # catch if char isn't used
    if idx <= 0:
        idx = prompt[:max_len].rfind(' ')

    return prompt[:idx]

# build pipeline
assert torch.cuda.is_available()
pipeline = DiffusionPipeline.from_pretrained('pipelines/SD-v1-5').to('cuda')
pipeline.enable_attention_slicing()
# disable safety checker
def dummy(images, **kwargs):
    return images, [False]
pipeline.safety_checker = dummy

# generate images
def gen_loop(pipeline, prompts, seeds, x, y, m_vol, num_imgs, dir, logger):    
    tomesd.apply_patch(pipeline, m_vol, merge_attn=False, merge_crossattn=True, merge_mlp=False)
    for i in range(num_imgs):
        prompt, seed = cut_prompt(prompts[i]), seeds[i].item()
        start = time.time()
        image = pipeline(prompt, x, y, generator=torch.Generator().manual_seed(seed)).images[0]
        end = time.time()
        diff_time = end - start

        name = 'img_' + ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        image.save(f'/gpfs/scratch/hebal100/{MAIN_DIR}/{dir}/{name}.png')
        # create new log entry
        logger.append([prompt, seed, m_vol, diff_time, name])

merge_volumes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
directories = ['images_0', 'images_10', 'images_20', 'images_30', 'images_40', 'images_50', 'images_60']
logger = []

for m_vol, dir in zip(merge_volumes, directories):
    gen_loop(pipeline, prompts, seeds, x, y, m_vol, sample_size, dir, logger)
tomesd.remove_patch(pipeline)

# save log
log = pd.DataFrame(logger, columns=['prompt', 'seed', 'm_vol', 'time', 'name'])
name = 'log_' + ''.join(random.choices(string.ascii_letters + string.digits, k=5))
log.to_csv(f'/gpfs/scratch/hebal100/{MAIN_DIR}/logger/{name}.csv', index=False)
