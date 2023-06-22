from diffusers import DiffusionPipeline
import sys

# python load_models.py model_id_from_huggingface model_name_to_save
model_id = sys.argv[1]

# Download files ahead of time
pipeline = DiffusionPipeline.from_pretrained(model_id)

# Save files to a specified directory
pipeline_path = "/Users/henri/Info/bachelorarbeit/pipelines/"+sys.argv[2]
pipeline.save_pretrained(pipeline_path)


