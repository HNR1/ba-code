from diffusers import DiffusionPipeline
import torch, os.path
import tomesd
import pytorch_fid.src.fid.fid_score as fid_score
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = DiffusionPipeline.from_pretrained("/gpfs/project/hebal100/ba-code/pipelines/SD-v1-5")
pipeline.to(device)
pipeline.enable_attention_slicing()

prompt = "photograph of an astronaut riding a horse"
seed = 555035

image_c1 = pipeline(prompt, 768, 768, generator=torch.Generator().manual_seed(seed)).images[0]
image_c2 = pipeline(prompt, 768, 768, generator=torch.Generator().manual_seed(seed)).images[0]
tomesd.apply_patch(pipeline, 0.5)
image_m1 = pipeline(prompt, 768, 768, generator=torch.Generator().manual_seed(seed)).images[0]
image_m2 = pipeline(prompt, 768, 768, generator=torch.Generator().manual_seed(seed)).images[0]
tomesd.remove_patch(pipeline)

fid1 = fid_score.calc_fid_given_lists(image_c1, image_c2)
fid2 = fid_score.calc_fid_given_lists(image_m1, image_m2)

print(fid1, fid2)
