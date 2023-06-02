import requests
from fastapi import FastAPI
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

app = FastAPI()


@app.get("/")
async def root():
    return {"introduction": "Get image caption from open source models"}

@app.get("/caption/")
async def getCaption(img_url: str, prompt: str = ''):
    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    
    if prompt:
        inputs = processor(image, prompt, return_tensors="pt").to("cuda")
    else:
        inputs = processor(image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True);

    return {
        "caption": caption
    }