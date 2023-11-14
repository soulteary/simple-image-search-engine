import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import time
import os
import redis

# 连接 Redis 数据库，地址换成你自己的 Redis 地址
client = redis.Redis(host="redis-server", port=6379, decode_responses=True)
res = client.ping()
print("redis connected:", res)

model_name_or_local_path = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name_or_local_path)
processor = CLIPProcessor.from_pretrained(model_name_or_local_path)

image_directory = "images"
png_files = [filename for filename in os.listdir(image_directory) if filename.endswith(".png")]
sorted_png_files = sorted(png_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))

# 初始化 Redis Pipeline
pipeline = client.pipeline()
for i, png_file in enumerate(sorted_png_files, start=1):
    # 初始化 Redis，先使用 PNG 文件名作为 Key 和 Value，后续再更新为图片特征向量
    pipeline.json().set(png_file, "$", png_file)

batch_size = 1

with torch.no_grad():
    for idx, png_file in enumerate(sorted_png_files, start=1):
        print(f"{idx}: {png_file}")
        start = time.time()
        image = Image.open(f"{image_directory}/{png_file}")
        inputs = processor(images=image, return_tensors="pt", padding=True)
        image_features = model.get_image_features(inputs.pixel_values)[batch_size-1]
        embeddings = image_features.numpy().astype(np.float32).tolist()
        print('image_features:', embeddings)
        vector_dimension = len(embeddings)
        print('vector_dimension:', vector_dimension)
        end = time.time()
        print('%s Seconds'%(end-start))
        # 更新 Redis 数据库中的文件向量
        pipeline.json().set(png_file, "$", embeddings)
        res = pipeline.execute()
        print('redis set:', res)