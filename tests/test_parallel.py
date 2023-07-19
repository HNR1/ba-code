"""
import sys       
sys.path.insert(1, '/gpfs/project/hebal100/ba-code/libs')
from diffusers import DiffusionPipeline
import torch, numpy as np
import torch.nn as nn
#from tomesd import tomesd
#from pytorch_fid.src.fid.fid_score import calc_fid_given_lists

assert torch.cuda.is_available()
pipeline = DiffusionPipeline.from_pretrained("/gpfs/project/hebal100/ba-code/pipelines/SD-v1-5").to('cuda')
pipeline.enable_attention_slicing()

seeds = np.random.randint(1, 2000000000, 5)

filename = "/gpfs/project/hebal100/ba-code/data/prompts.txt"
prompts = []
with open(filename, 'r', encoding='UTF-8') as file:
    while line := file.readline():
        prompts.append(line.rstrip())

x, y = 768, 768
images = []
images_tome = []

num_images = 5

for i in range(num_images):
    images.append(pipeline(prompts[i], x, y, generator=torch.Generator().manual_seed(seeds[i])).images[0])

print(len(images))
#tomesd.apply_patch(pipeline, 0.5)

#tomesd.remove_patch(pipeline)

#fid = calc_fid_given_lists(images, images_tome)

#print(fid)
"""
#
#
#
#
#
#
import sys
sys.path.insert(1, '/gpfs/project/hebal100/ba-code/libs')
from diffusers import DiffusionPipeline
import torch, numpy as np
import torch.nn as nn

assert torch.cuda.is_available()
print(torch.cuda.device_count())
num_gpus = torch.cuda.device_count()
if num_gpus < 1:
    raise RuntimeError("No GPUs available for parallel execution.")

pipeline = DiffusionPipeline.from_pretrained("/gpfs/project/hebal100/ba-code/pipelines/SD-v1-5")

# Enable attention slicing for parallel execution
pipeline.enable_attention_slicing()

seeds = np.random.randint(1, 2000000000, num_gpus)

filename = "/gpfs/project/hebal100/ba-code/data/prompts.txt"
prompts = []
with open(filename, 'r', encoding='UTF-8') as file:
    while line := file.readline():
        prompts.append(line.rstrip())

x, y = 768, 768
images = []

for i in range(num_gpus):
    # Set the current GPU
    torch.cuda.set_device(i)

    # Move the pipeline to the current GPU
    pipeline.to(torch.device('cuda', i))

    # Wrap the pipeline with DataParallel
    pipeline_parallel = nn.DataParallel(pipeline)

    # Generate image on the current GPU
    image = pipeline_parallel(prompts[i], x, y, generator=torch.Generator().manual_seed(seeds[i])).images[0]

    images.append(image)

print(len(images))
