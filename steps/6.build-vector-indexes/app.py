import redis
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# 连接 Redis 数据库，地址换成你自己的 Redis 地址
client = redis.Redis(host="redis-server", port=6379, decode_responses=True)
res = client.ping()
print("redis connected:", res)

# 之前模型处理的向量维度是 512
vector_dimension = 512
# 给索引起个与众不同的名字
vector_indexes_name = "idx:ball_indexes"

# 定义向量数据库的 Schema
schema = (
    VectorField(
        "$",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": vector_dimension,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="vector",
    ),
)
# 设置一个前缀，方便后续查询，也作为命名空间和可能的普通数据进行隔离
# 这里设置为 ball-，未来可以通过 ball-* 来查询所有数据
definition = IndexDefinition(prefix=["ball-"], index_type=IndexType.JSON)
# 使用 Redis 客户端实例根据上面的 Schema 和定义创建索引
res = client.ft(vector_indexes_name).create_index(
    fields=schema, definition=definition
)
print("create_index:", res)
