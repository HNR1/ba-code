from diffusers import DiffusionPipeline
import torch
from libs.tomesd import tomesd
from libs.pytorch_fid.src.fid.fid_score import calc_fid_given_lists

assert torch.cuda.is_available()
pipeline = DiffusionPipeline.from_pretrained("/gpfs/project/hebal100/ba-code/pipelines/SD-v1-5").to('cuda')
pipeline.enable_attention_slicing()

prompt1 = "photograph of an astronaut riding a horse"
seed1 = 555035
prompt2 = "cowboy"
seed2 = 783047

image_c1 = pipeline(prompt1, 768, 768, generator=torch.Generator(device='cuda').manual_seed(seed1)).images[0]
image_c2 = pipeline(prompt2, 768, 768, generator=torch.Generator(device='cuda').manual_seed(seed2)).images[0]
tomesd.apply_patch(pipeline, 0.5)
image_m1 = pipeline(prompt1, 768, 768, generator=torch.Generator(device='cuda').manual_seed(seed1)).images[0]
image_m2 = pipeline(prompt2, 768, 768, generator=torch.Generator(device='cuda').manual_seed(seed2)).images[0]
tomesd.remove_patch(pipeline)

fid1 = calc_fid_given_lists(image_c1, image_m1)
fid2 = calc_fid_given_lists(image_c2, image_m2)
fid3 = calc_fid_given_lists([image_c1, image_c2], [image_m1, image_m2])
fid4 = (fid1+fid2)/2

print(fid1, fid2, fid3, fid4)