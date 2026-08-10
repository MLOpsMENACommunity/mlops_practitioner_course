# consumer.py — event-driven inference consumer
import json, mlflow.sklearn, os

MODEL = None  # loaded once on cold start, cached in memory

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = mlflow.sklearn.load_model(os.environ["MODEL_URI"])
    return MODEL

def store_result(result: dict) -> None:
    """Simplest sink: append to a JSONL file. Swap for PostgreSQL, Redis, etc."""
    with open("predictions.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

def predict_on_event(event: dict) -> dict:
    """Called for every incoming message from the broker."""
    model   = get_model()
    payload = json.loads(event["data"])
    # must match src/train.py FEATURES = ["distance_km", "passengers"]
    features = [payload["distance_km"], payload["passengers"]]
    prediction = model.predict([features])[0]
    result = {"ride_id": payload["ride_id"], "prediction": round(float(prediction), 2)}
    store_result(result)  # PostgreSQL, Redis, file, etc.
    return result

# Run as a long-running consumer process:
#   docker run -d --name kafka -p 9092:9092 apache/kafka   # single-node KRaft broker
#   export MLFLOW_TRACKING_URI="http://localhost:5000"   # registry stores mlflow-artifacts:/ URIs
#   export MODEL_URI="models:/RideDurationModel@production"
#   python consumer.py                                     # blocks, handling messages forever
# or wrap in Docker with --restart unless-stopped
#
# Publish a test event from another shell:
#   docker exec -i kafka /opt/kafka/bin/kafka-console-producer.sh \
#     --bootstrap-server localhost:9092 --topic ride-events <<< \
#     '{"ride_id": "r1", "distance_km": 3.2, "passengers": 1}'

if __name__ == "__main__":
    from confluent_kafka import Consumer

    topic = os.environ.get("RIDE_EVENTS_TOPIC", "ride-events")
    consumer = Consumer({
        "bootstrap.servers": os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092"),
        "group.id": "ride-duration-consumer",
        "auto.offset.reset": "earliest",  # new consumer group starts from the oldest event
    })
    consumer.subscribe([topic])
    print(f"consumer listening on '{topic}' — Ctrl+C to stop")

    try:
        while True:
            message = consumer.poll(timeout=1.0)
            if message is None:
                continue
            if message.error():
                print("broker error:", message.error())
                continue
            try:
                result = predict_on_event({"data": message.value().decode("utf-8")})
                print("processed:", result)
            except Exception as exc:
                # a bad message must not kill the consumer — log and move on
                print("skipped malformed event:", exc)
    finally:
        consumer.close()  # commits final offsets and leaves the group cleanly
