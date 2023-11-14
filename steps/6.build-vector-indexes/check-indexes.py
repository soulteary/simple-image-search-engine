import redis

# 连接 Redis 数据库，地址换成你自己的 Redis 地址
client = redis.Redis(host="redis-server", port=6379, decode_responses=True)
res = client.ping()
print("redis connected:", res)

vector_indexes_name = "idx:ball_vss"

# 从 Redis 数据库中读取索引状态
info = client.ft(vector_indexes_name).info()
# 获取索引状态中的 num_docs 和 hash_indexing_failures
num_docs = info["num_docs"]
indexing_failures = info["hash_indexing_failures"]
print(f"{num_docs} documents indexed with {indexing_failures} failures")
