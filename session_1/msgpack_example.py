import json
import msgpack

data = {"distance_km": 5.2, "passengers": 2, "model_version": "v1.3"}

# JSON — text format
json_bytes = json.dumps(data).encode()
print(len(json_bytes))  # 54 bytes
print(json_bytes)  # b'{"distance_km": 5.2, "passengers": 2, ...}'

# MessagePack — binary format
msgpack_bytes = msgpack.packb(data)
print(len(msgpack_bytes))  # 38 bytes  (~30% smaller)
print(msgpack_bytes)  # b'\x83\xabdistance_km...'  (unreadable, but smaller)

# _________________________________ #

import redis  # noqa: E402  (imported here to keep this Redis example self-contained)
import msgpack  # noqa: E402

r = redis.Redis()

# Writing features to Redis with MessagePack
features = {"user_age": 28, "purchase_history": [12.5, 8.0, 45.2], "last_login_days": 3}
r.set("user:1234:features", msgpack.packb(features))

# Reading and deserializing for prediction
raw = r.get("user:1234:features")
features = msgpack.unpackb(raw, raw=False)
# prediction = model.predict([features["user_age"], features["last_login_days"]])
