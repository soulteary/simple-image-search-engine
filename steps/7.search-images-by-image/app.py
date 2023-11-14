import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import time
import redis
from redis.commands.search.query import Query

model_name_or_local_path = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name_or_local_path)
processor = CLIPProcessor.from_pretrained(model_name_or_local_path)

vector_indexes_name = "idx:ball_indexes"

client = redis.Redis(host="10.11.12.178", port=6379, decode_responses=True)
res = client.ping()
print("redis connected:", res)

start = time.time()
image = Image.open("ball-8576.png")
batch_size = 1

with torch.no_grad():
    inputs = processor(images=image, return_tensors="pt", padding=True)
    image_features = model.get_image_features(inputs.pixel_values)[batch_size-1]
    embeddings = image_features.numpy().astype(np.float32).tobytes()
    print('image_features:', embeddings)

# 构建请求命令，查找和我们提供图片最相近的 30 张图片
query_vector = embeddings
query = (
    Query("(*)=>[KNN 30 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("$")
    .dialect(2)
)

# 定义一个查询函数，将我们查找的结果的 ID 打印出来（图片名称）
def dump_query(query, query_vector, extra_params={}):
    result_docs = (
        client.ft(vector_indexes_name)
        .search(
            query,
            {
                "query_vector": query_vector
            }
            | extra_params,
        )
        .docs
    )
    print(result_docs)
    for doc in result_docs:
        print(doc['id'])

dump_query(query, query_vector, {})

end = time.time()
print('%s Seconds'%(end-start))
