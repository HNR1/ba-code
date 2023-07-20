import sys, os
sys.path.insert(1, '/gpfs/project/hebal100/ba-code/libs')
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8,9'
import torch, numpy as np
from diffusers import StableDiffusionParadigmsPipeline

assert torch.cuda.is_available()
pipeline = StableDiffusionParadigmsPipeline.from_pretrained("/gpfs/project/hebal100/ba-code/pipelines/SD-v1-5-parallel").to('cuda')
pipeline.enable_attention_slicing()

ngpu, batch_per_device = 10, 5

pipeline.wrapped_unet = torch.nn.DistributedDataParallel(pipeline.unet, device_ids=[0,1,2,3,4,5,6,7,8,9])

filename = "/gpfs/project/hebal100/ba-code/data/prompts.txt"
prompts = []
with open(filename, 'r', encoding='UTF-8') as file:
    while line := file.readline():
        prompts.append(line.rstrip())
seeds = np.random.randint(1, 2000000000, len(prompts))

x, y = 768, 768
images = []

for i in range(len(prompts)):

    image = pipeline(prompts[i], x, y, parallel=ngpu * batch_per_device,
                     generator=torch.Generator().manual_seed(seeds[i].item())).images[0]

    images.append(image)

print(len(images))