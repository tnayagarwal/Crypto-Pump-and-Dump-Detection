import json
import time
import random
from kafka import KafkaProducer

def get_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def simulate_stream():
    """Simulates high-frequency OHLCV tick data for crypto markets."""
    producer = get_producer()
    topic = 'crypto_ticks'
    
    print(f"Starting mock crypto stream to topic: {topic}")
    try:
        while True:
            # Mock tick data
            tick = {
                "timestamp": int(time.time()),
                "token_id": random.choice(["TOKEN_A", "TOKEN_B", "TOKEN_C"]),
                "price_usd": round(random.uniform(0.01, 1000.0), 4),
                "volume_24h": round(random.uniform(10000, 5000000), 2),
                "buy_sell_ratio": round(random.uniform(0.1, 9.9), 2)
            }
            producer.send(topic, tick)
            print(f"Produced: {tick}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stream interrupted.")

if __name__ == "__main__":
    simulate_stream()
