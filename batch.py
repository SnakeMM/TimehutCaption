import sys
import pandas as pd
import httpx
import time
import torch
import asyncio
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

baby_id = '537592620'
total_count = 0
current_count = 0
multi_count = int(sys.argv[1])
download_times = []
compress_times = []
inference_times = []

def readUrls():
    image_urls = []
    df = pd.read_csv(f'{baby_id}.csv')
    for index, row in df.iterrows():
        img_url = row[3]
        if isinstance(img_url, str) and img_url.startswith("http") and not img_url.endswith("heic"):
            image_urls.append(img_url)
            print(f"{index} Download url:", img_url)
    
    run_all_tasks(image_urls[:1000])

def run_all_tasks(image_urls):
    global total_count
    global current_count
    total_count = len(image_urls)
    current_count = 0
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(run_task(image_urls))

async def run_task(image_urls):
    #tasks = [download_and_inference_image(url) for url in image_urls]
    #results = await asyncio.gather(*tasks)
    #return results

    t1 = time.perf_counter()

    semaphore = asyncio.Semaphore(multi_count)
    async def download_with_sem(url):
        async with semaphore:
            await download_and_inference_image(url)

    tasks = [download_with_sem(url) for url in image_urls]
    await asyncio.gather(*tasks)

    t2 = time.perf_counter()
    print(f"run task success {int((t2 - t1) * 1000)} ms")
    sum_d = sum(download_times)
    len_d = len(download_times)
    print(f"download: {len_d} average {int(sum_d / len_d)} ms")
    sum_c = sum(compress_times)
    len_c = len(compress_times)
    print(f"compress: {len_c} average {int(sum_c / len_c)} ms")
    sum_i = sum(inference_times)
    len_i = len(inference_times)
    print(f"inference: {len_i} average {int(sum_i / len_i)} ms")

async def download_and_inference_image(img_url):
    global current_count
    global total_count
    t1 = time.perf_counter()
    image_data = await download_image(img_url)
    if image_data:
        t2 = time.perf_counter()
        td = int((t2 - t1) * 1000)
        download_times.append(td)
        print(f"download success {td} ms")
        image = await compress_image(image_data)
        if image:
            t3 = time.perf_counter()
            tc = int((t3 - t2) * 1000)
            compress_times.append(tc)
            print(f"compress success {tc} ms")
            inputs = processor(images=image, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)
            image_feature = image_features[0].detach().cpu().numpy()
            if (len(image_feature) == 512):
                current_count += 1
                t4 = time.perf_counter()
                ti = int((t4 - t3) * 1000)
                inference_times.append(ti)
                print(f"{current_count} inference success {ti} ms")
        else:
            current_count += 1
            print(f"compress fail: {img_url}")
    else:
        current_count += 1
        print(f"download fail: {img_url}")

async def download_image(img_url):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(img_url)
            if response.status_code == 200:
                return response.content
            else:
                return None
        except Exception as e:
            print(f"Download exception: {e}")
            return None
        
async def compress_image(image):
    try:
        t1 = time.perf_counter()
        image = Image.open(BytesIO(image)).convert('RGB')
        t2 = time.perf_counter()
        #print((t2 - t1) * 1000)

        # 居中裁剪图像
        width, height = image.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        image_cropped = image.crop((left, top, right, bottom))

        t3 = time.perf_counter()
        #print((t3 - t2) * 1000)

        # 缩放图像至 224x224
        image_resized = image_cropped.resize((224, 224))

        t4 = time.perf_counter()
        #print((t4 - t3) * 1000)
        return image_resized
    except Exception as e:
        print(f"Compress exception: {e}")
        return None

def main():
    readUrls()
 
if __name__ == "__main__":
    main()