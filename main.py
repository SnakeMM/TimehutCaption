import os
import requests
from fastapi import FastAPI
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv

load_dotenv()

useCpu = os.getenv("PROCESSING_UNIT") == "CPU"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
if useCpu:
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
else:
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

app = FastAPI()


@app.get("/")
async def root():
    return {"introduction": "Get image caption from open source models"}

@app.get("/caption/")
async def getCaption(img_url: str, prompt: str = ''):
    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    
    if prompt:
        if useCpu:
            inputs = processor(image, prompt, return_tensors="pt")
        else:
            inputs = processor(image, prompt, return_tensors="pt").to("cuda")
    else:
        if useCpu:
            inputs = processor(image, return_tensors="pt")
        else:
            inputs = processor(image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True);

    return {
        "caption": caption
    }