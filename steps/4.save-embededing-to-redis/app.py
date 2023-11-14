import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import time
import redis

# 连接 Redis 数据库，地址换成你自己的 Redis 地址
client = redis.Redis(host="redis-server", port=6379, decode_responses=True)
res = client.ping()
print("redis connected:", res)

model_name_or_local_path = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name_or_local_path)
processor = CLIPProcessor.from_pretrained(model_name_or_local_path)

png_file = "ball-8576.png"

start = time.time()
image = Image.open(png_file)
batch_size = 1

# 初始化 Redis Pipeline
pipeline = client.pipeline()
# 初始化 Redis，先使用 PNG 文件名作为 Key 和 Value，后续再更新为图片特征向量
pipeline.json().set(png_file, "$", png_file)
res = pipeline.execute()
print('redis set keys:', res)

with torch.no_grad():
    inputs = processor(images=image, return_tensors="pt", padding=True)
    image_features = model.get_image_features(inputs.pixel_values)[batch_size-1]
    embeddings = image_features.numpy().astype(np.float32).tolist()
    vector_dimension = len(embeddings)
    print('vector_dimension:', vector_dimension)
    end = time.time()
    print('%s Seconds'%(end-start))
    # 将计算出的 Embeddings 更新到 Redis 数据库中
    pipeline.json().set(png_file, "$", embeddings)
    res = pipeline.execute()
    print('redis set:', res)
