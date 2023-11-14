import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import time

# 默认从 HuggingFace 加载模型，也可以从本地加载，需要提前下载完毕
model_name_or_local_path = "openai/clip-vit-base-patch16"

model = CLIPModel.from_pretrained(model_name_or_local_path)
processor = CLIPProcessor.from_pretrained(model_name_or_local_path)

# 记录处理开始时间
start = time.time()
# 读取待处理图片
image = Image.open("ball-8576.png")
# 处理图片数量，这里每次只处理一张图片
batch_size = 1

with torch.no_grad():
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
