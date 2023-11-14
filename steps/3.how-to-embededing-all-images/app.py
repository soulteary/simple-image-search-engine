import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import time
import os

# 默认从 HuggingFace 加载模型，也可以从本地加载，需要提前下载完毕
model_name_or_local_path = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name_or_local_path)
processor = CLIPProcessor.from_pretrained(model_name_or_local_path)

image_directory = "images"
# 使用列表推导式获取目录中所有的 PNG 图片名称
png_files = [filename for filename in os.listdir(image_directory) if filename.endswith(".png")]
# 根据文件名中的数字部分进行排序
sorted_png_files = sorted(png_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))

# 处理图片数量，这里每次只处理一张图片
batch_size = 1

with torch.no_grad():
    # 打印排序后的 PNG 图片名称列表
    for idx, png_file in enumerate(sorted_png_files, start=1):
        print(f"{idx}: {png_file}")
        # 记录处理开始时间
        start = time.time()
        # 读取待处理图片
        image = Image.open(f"{image_directory}/{png_file}")
        # 将图片使用模型加载，转换为 PyTorch 的 Tensor 数据类型
        # 你也可以在这里对图片进行一些特殊处理，裁切、缩放、超分、重新取样等等
        inputs = processor(images=image, return_tensors="pt", padding=True)
        # 使用模型处理图片的 Tensor 数据，获取图片特征向量
        image_features = model.get_image_features(inputs.pixel_values)[batch_size-1]
        # 将图片特征向量转换为 Numpy 数组，未来可以存储到数据库中
        embeddings = image_features.numpy().astype(np.float32).tolist()
        print('image_features:', embeddings)
        # 打印向量维度，这里是 512 维
        vector_dimension = len(embeddings)
        print('vector_dimension:', vector_dimension)
        # 计算整个处理过程的时间
        end = time.time()
        print('%s Seconds'%(end-start))
