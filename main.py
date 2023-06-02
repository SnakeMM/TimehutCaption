import os
import requests
from fastapi import FastAPI
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from dotenv import load_dotenv

load_dotenv()

useCpu = os.getenv("PROCESSING_UNIT") == "CPU"

# BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
if useCpu:
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
else:
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

# BLIP2
processor2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
if useCpu:
    model2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
else:
    model2 = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

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