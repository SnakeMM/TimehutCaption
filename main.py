import os
import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel

tags = ["man","woman","dad","mom","boy","girl","baby","dog","cat","beach"]

device = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()

useBLIP = os.getenv("MODEL_BLIP_ENABLED") == "True"
useBLIP2 = os.getenv("MODEL_BLIP2_ENABLED") == "True"
useCLIP = os.getenv("MODEL_CLIP_ENABLED") == "True"

# BLIP
if (useBLIP):
    processor1 = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model1 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# BLIP2
if (useBLIP2):
    processor2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

# CLIP
if (useCLIP):
    processor3 = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model3 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

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

    if not useBLIP:
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

    if not useBLIP2:
        return {"error": "model disabled"}

    if prompt:
        inputs = processor2(image, prompt, return_tensors="pt").to(device, torch.float16)
    else:
        inputs = processor2(image, return_tensors="pt").to(device, torch.float16)

    out = model2.generate(**inputs)
    caption = processor2.decode(out[0], skip_special_tokens=True).strip();

    return {
        "caption": caption
    }

@app.get("/tags/clip/")
async def getTagsClip(
    img_url: str, 
):
    if not useCLIP:
        return {"error": "model disabled"}

    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    inputs = processor3(text=tags, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model3(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]

    results = []
    for index,tag in enumerate(tags):
        if probs[index] >= 0.01:
            results.append({
                "name": tag,
                "confidence": probs[index]
            })
    results = sorted(results, key=lambda k: k["confidence"], reverse=True)

    return {
        "model": "CLIP",
        "tags": results
    }