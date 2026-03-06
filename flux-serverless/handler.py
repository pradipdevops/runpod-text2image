import runpod
import torch
from diffusers import FluxPipeline
from io import BytesIO

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)

pipe = pipe.to("cuda")

def handler(event):
    prompt = event["input"].get("prompt", "A lion sitting on a throne")

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=30,
    ).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "image/png",
            "Content-Disposition": "attachment; filename=flux.png"
        },
        "body": img_bytes,
        "isBase64Encoded": False
    }

runpod.serverless.start({"handler": handler})
