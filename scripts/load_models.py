from diffusers import DiffusionPipeline, DDPMParallelScheduler, StableDiffusionParadigmsPipeline
import sys, torch

# python load_models.py model_id_from_huggingface model_name_to_save
model_id = sys.argv[1]

# Download files ahead of time
#pipeline = DiffusionPipeline.from_pretrained(model_id)
scheduler = DDPMParallelScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionParadigmsPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)


# Save files to a specified directory
pipeline_path = "/Users/henri/Info/ba-code/pipelines/"+sys.argv[2]
pipe.save_pretrained(pipeline_path)


