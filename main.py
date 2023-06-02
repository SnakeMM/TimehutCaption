import os
import requests
import torch
from fastapi import FastAPI
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from dotenv import load_dotenv

device = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()

useBIP = os.getenv("MODEL_BIP_ENABLED") == "True"
useBIP2 = os.getenv("MODEL_BIP2_ENABLED") == "True"

# BLIP
if (useBIP):
    processor1 = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model1 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# BLIP2
if (useBIP2):
    processor2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

app = FastAPI()

@app.get("/")
async def root():
    return {"introduction": "Get image caption from open source models"}

@app.get("/caption/blip/")
async def getCaptionBlip(
    img_url: str, 
    prompt: str = '',
):
    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    if not useBIP:
        return {"error": "model disabled"}

    if prompt:
        inputs = processor1(image, prompt, return_tensors="pt").to(device)
    else:
        inputs = processor1(image, return_tensors="pt").to(device)

    out = model1.generate(**inputs)
    caption = processor1.decode(out[0], skip_special_tokens=True);

    return {
        "caption": caption
    }

@app.get("/caption/blip2/")
async def getCaptionBlip2(
    img_url: str, 
    prompt: str = '',
):
    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    if not useBIP2:
        return {"error": "model disabled"}

    if prompt:
        inputs = processor2(image, prompt, return_tensors="pt").to(device, torch.float16)
    else:
        inputs = processor2(image, return_tensors="pt").to(device, torch.float16)

    out = model2.generate(**inputs)
    caption = processor2.decode(out[0], skip_special_tokens=True);

    return {
        "caption": caption
    }