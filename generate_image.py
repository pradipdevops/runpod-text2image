from diffusers import StableDiffusionPipeline
import sys

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

pipe = pipe.to("cpu")

prompt = sys.argv[1] if len(sys.argv) > 1 else "Lion"

image = pipe(prompt).images[0]

image.save("output.png")

print("Saved as output.png")
