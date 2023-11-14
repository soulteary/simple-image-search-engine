import redis

# 连接 Redis 数据库，地址换成你自己的 Redis 地址
client = redis.Redis(host="redis-server", port=6379, decode_responses=True)
res = client.ping()
print("redis connected:", res)

res = client.json().get("ball-1234.png")
print(res)