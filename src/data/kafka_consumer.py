import json
from kafka import KafkaConsumer

def get_consumer(topic: str) -> KafkaConsumer:
    return KafkaConsumer(
        topic,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

def consume_stream():
    """Consumes real-time crypto streams and passes them to the GNN anomaly detector."""
    topic = 'crypto_ticks'
    consumer = get_consumer(topic)
    
    print(f"Subscribed to {topic}. Waiting for ticks...")
    try:
        for message in consumer:
            tick = message.value
            print(f"Evaluating: {tick['token_id']} @ ${tick['price_usd']}")
            # Future: route to src.models.gnn_lstm for real-time anomaly inference
    except KeyboardInterrupt:
        print("Consumer interrupted.")

if __name__ == "__main__":
    consume_stream()
