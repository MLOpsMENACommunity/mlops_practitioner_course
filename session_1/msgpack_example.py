import json
import msgpack

data = {"distance_km": 5.2, "passengers": 2, "model_version": "v1.3"}

# JSON — text format
json_bytes = json.dumps(data).encode()
print(len(json_bytes))  # 62 bytes
print(json_bytes)  # b'{"distance_km": 5.2, "passengers": 2, ...}'

# MessagePack — binary format
msgpack_bytes = msgpack.packb(data)
print(len(msgpack_bytes))  # 53 bytes  (~15% smaller)
print(msgpack_bytes)  # b'\x83\xabdistance_km...'  (unreadable, but smaller)

# _________________________________ #

# Everything above is the actual serialization lesson — it needs nothing but
# `msgpack`. The optional demo below shows the same encoding used as a feature
# cache payload; it needs the `redis` package *and* a running server, and is
# skipped cleanly when either is missing.


def redis_cache_demo() -> None:
    """Store and read back a feature dict in Redis, MessagePack-encoded."""
    import redis

    r = redis.Redis()

    features = {
        "user_age": 28,
        "purchase_history": [12.5, 8.0, 45.2],
        "last_login_days": 3,
    }

    try:
        # Writing features to Redis with MessagePack
        r.set("user:1234:features", msgpack.packb(features))

        # Reading and deserializing for prediction
        raw = r.get("user:1234:features")
        features = msgpack.unpackb(raw, raw=False)
        print(features)
        # prediction = model.predict([features["user_age"], features["last_login_days"]])
    except redis.ConnectionError:
        print("\nRedis is not reachable on localhost:6379 — skipping the cache demo.")
        print("Start one with:  docker run -d -p 6379:6379 redis:7-alpine")


try:
    redis_cache_demo()
except ImportError:
    print("\n`redis` is not installed — skipping the cache demo.")
    print('Install it with:  pip install -e ".[msgpack,redis]"')
