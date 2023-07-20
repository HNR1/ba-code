import sys, os
sys.path.insert(1, '/gpfs/project/hebal100/ba-code/libs')
import torch, numpy as np
from diffusers import StableDiffusionParadigmsPipeline
from torch.nn.parallel import DistributedDataParallel as DDP
from tomesd import tomesd

assert torch.cuda.is_available()
pipeline = StableDiffusionParadigmsPipeline.from_pretrained("/gpfs/project/hebal100/ba-code/pipelines/SD-v1-5-parallel").to('cuda')
pipeline.enable_attention_slicing()

ngpu, batch_per_device = min(8, torch.cuda.device_count()), 6

#torch.distributed.init_process_group(backend='nccl', world_size=ngpu, rank=0, world_size=1)
#pipeline.wrapped_unet = DDP(pipeline.unet)
pipeline.wrapped_unet = torch.nn.DataParallel(pipeline.unet, device_ids=[d for d in range(ngpu)])

filename = "/gpfs/project/hebal100/ba-code/data/prompts.txt"
prompts = []
with open(filename, 'r', encoding='UTF-8') as file:
    while line := file.readline():
        prompts.append(line.rstrip())
seeds = np.random.randint(0, 4294967295, len(prompts))

x, y = 768, 768
images = []
images_tome = []

num_imgs = 2
#num_imgs = len(prompts)
for i in range(num_imgs):
    torch.cuda.empty_cache()
    image = pipeline(prompts[i], x, y, parallel=ngpu * batch_per_device,
                     generator=torch.Generator().manual_seed(seeds[i].item())).images[0]

    images.append(image)

tomesd.apply_patch(pipeline, 0.5)

for i in range(num_imgs):
    torch.cuda.empty_cache()
    image = pipeline(prompts[i], x, y, parallel=ngpu * batch_per_device,
                     generator=torch.Generator().manual_seed(seeds[i].item())).images[0]

    images_tome.append(image)

tomesd.remove_patch(pipeline)

print(len(images))