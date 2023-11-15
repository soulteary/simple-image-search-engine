import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import time

# 默认从 HuggingFace 加载模型，也可以从本地加载，需要提前下载完毕
model_name_or_local_path = "openai/clip-vit-base-patch16"
# 加载模型
model = CLIPModel.from_pretrained(model_name_or_local_path)
processor = CLIPProcessor.from_pretrained(model_name_or_local_path)

# 记录处理开始时间
start = time.time()
# 读取待处理图片
image = Image.open("ball-8576.png")
# 处理图片数量，这里每次只处理一张图片
batch_size = 1

# 要检测是否在图片中出现的内容
text = ['dog', 'cat', 'night', 'astronaut']

with torch.no_grad():
    # 将图片使用模型加载，转换为 PyTorch 的 Tensor 数据类型
    # 相比较第一篇文章中的例子 1.how-to-embededing/app.py，这里多了一个 text 参数
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    # 将 inputs 中的内容解包，传递给模型，调用模型处理图片和文本
    outputs = model(**inputs)
    # 将原始模型输出转换为类别概率分布（在类别维度上执行 softmax 激活函数）
    probs = outputs.logits_per_image.softmax(dim=1)
    end = time.time()
    # 记录处理结束时间
    print('%s Seconds'%(end-start))
    # 打印所有的概率分布
    for i in range(len(text)):
        print(text[i],":",probs[0][i])