import sys       
sys.path.insert(1, '/gpfs/project/hebal100/ba-code')
from diffusers import DiffusionPipeline
import torch
from libs.tomesd import tomesd

assert torch.cuda.is_available()
pipeline = DiffusionPipeline.from_pretrained("/gpfs/project/hebal100/ba-code/pipelines/SD-v1-5").to('cuda')
pipeline.enable_attention_slicing()

prompt = "photograph of an astronaut riding a horse"
seed = 555035

x, y = 768, 768
tomesd.apply_patch(pipeline, 0.5, use_rand=False)
image_m1 = pipeline(prompt, x, y, generator=torch.Generator().manual_seed(seed)).images[0]
tomesd.remove_patch(pipeline)