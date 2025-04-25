import os
import datetime
import requests
import torch
import tags_whitelist
from dotenv import load_dotenv
from fastapi import FastAPI
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from aesthetic_score_predictor import predict

# load image tags
tags = [] 
with open('tags' ,'r') as f:
    for line in f:
        tags.append(line.strip())

tags_all = [] 
with open('tags_all' ,'r') as f:
    for line in f:
        tags_all.append(line.strip())

device = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()

useBLIP = os.getenv("MODEL_BLIP_ENABLED") == "False"
useBLIP2 = os.getenv("MODEL_BLIP2_ENABLED") == "False"
useCLIP = True

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
    tokenizer3 = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

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
    img_tags: str = '', # split by comma
    threshold: float = 0.01
):
    if not useCLIP:
        return {"error": "model disabled"}

    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    if img_tags:
        input_tags = img_tags.split(',')
    else:
        input_tags = tags_all

    inputs = processor3(text=input_tags, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model3(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]

    results = []
    for index,tag in enumerate(input_tags):
        if probs[index] >= threshold:
            results.append({
                "name": tag,
                "confidence": probs[index]
            })
    results = sorted(results, key=lambda k: k["confidence"], reverse=True)

    return {
        "model": "CLIP",
        "tags": results[:10]
    }

@app.get("/analysis/")
async def getAnalysis(
    img_url: str
):
    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    t1 = datetime.datetime.now().microsecond

    tag1,prob1 = getMaxTag(image, tags_whitelist.tags_where_general)
    if tag1 == "indoor":
        tag2,prob2 = getMaxTag(image, tags_whitelist.tags_where_indoor_detail)
    else:
        tag2,prob2 = getMaxTag(image, tags_whitelist.tags_where_outdoor_detail)

    tag3,prob3 = getMaxTag(image, tags_whitelist.tags_what)

    #tag4,prob4 = getMaxTag(image, tags_whitelist.tags_people_count)

    t2 = datetime.datetime.now().microsecond

    return {
        #"people_count": tag4 + "_" + "%.2f" % prob4,
        "where": {
            "general": tag1 + "_" + "%.2f" % prob1,
            "detail": tag2 + "_" + "%.2f" % prob2
        },
        "what": {
            "detail": tag3 + "_" + "%.2f" % prob3
        },
        "time": int((t2 - t1) / 1000)
    }

def getMaxTag(image, input_tags):
    inputs = processor3(text=input_tags, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model3(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]
    max_prob = 0
    max_index = 0
    for index,prob in enumerate(probs):
        if prob > max_prob:
            max_prob = prob
            max_index = index

    return input_tags[max_index], max_prob

@app.get("/features/text/")
async def getTextFeatures(
    text: str
):
    inputs = tokenizer3([text], padding=True, return_tensors="pt").to(device)
    text_features = model3.get_text_features(**inputs)
    text_feature = text_features[0].detach().cpu().numpy()
    print(len(text_feature))
    
    return {
        "feature": text_feature.tolist()
    }

@app.get("/features/image/")
async def getImageFeatures(
    img_url: str
):
    t1 = datetime.datetime.now().microsecond

    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    t2 = datetime.datetime.now().microsecond

    inputs = processor3(images=image, return_tensors="pt").to(device)
    image_features = model3.get_image_features(**inputs)
    image_feature = image_features[0].detach().cpu().numpy()

    t3 = datetime.datetime.now().microsecond

    print(len(image_feature))
    
    return {
        "feature": image_feature.tolist(),
        "download_time": int((t2 - t1) / 1000),
        "calculate_time": int((t3 - t2) / 1000)
    }

@app.get("/score/")
async def getAestheticScore(
    img_url: str
):
    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    score = predict(image)

    return {
        "aesthetic score": score
    }