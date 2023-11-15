import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import time
import redis
from redis.commands.search.query import Query

model_name_or_local_path = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name_or_local_path)
processor = CLIPProcessor.from_pretrained(model_name_or_local_path)
# 处理文本需要引入
tokenizer = CLIPTokenizer.from_pretrained(model_name_or_local_path)

vector_indexes_name = "idx:ball_indexes"

client = redis.Redis(host="redis-server", port=6379, decode_responses=True)
res = client.ping()
print("redis connected:", res)

start = time.time()

# 调用模型获取文本的 embeddings
def get_text_embedding(text): 
    inputs = tokenizer(text, return_tensors = "pt")
    text_embeddings = model.get_text_features(**inputs)
    embedding_as_np = text_embeddings.cpu().detach().numpy()
    embeddings = embedding_as_np.astype(np.float32).tobytes()
    return embeddings

with torch.no_grad():
    # 获取文本的 embeddings
    text_embeddings = get_text_embedding('astronaut')

query_vector = text_embeddings
query = (
    Query("(*)=>[KNN 30 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("$")
    .dialect(2)
)

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
