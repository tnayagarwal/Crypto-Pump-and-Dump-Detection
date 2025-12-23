"""
Kafka Producer - Crypto Tick Stream Simulator
==============================================
Simulates high-frequency OHLCV cryptocurrency market data and
publishes it to a Kafka topic for real-time downstream processing.
"""

import json
import time
import random
import logging
from dataclasses import dataclass, asdict
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "crypto_ticks"
MOCK_TOKENS = ["TOKEN_A", "TOKEN_B", "TOKEN_C", "TOKEN_D"]
PRODUCER_CONFIG = {
    "retries": 5,
    "retry_backoff_ms": 300,
}


@dataclass
class TickEvent:
    timestamp: int
    token_id: str
    price_usd: float
    volume_24h: float
    buy_sell_ratio: float


def build_producer(max_retries: int = 5) -> KafkaProducer:
    """Create a KafkaProducer with retry logic for broker availability."""
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BOOTSTRAP],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                **PRODUCER_CONFIG,
            )
            logger.info("Kafka producer connected successfully.")
            return producer
        except NoBrokersAvailable:
            logger.warning(f"Broker unavailable. Retry {attempt + 1}/{max_retries}...")
            time.sleep(2 ** attempt)
    raise RuntimeError("Could not connect to Kafka broker after retries.")


def simulate_stream(ticks: int = -1) -> None:
    """
    Publish synthetic cryptocurrency tick data to Kafka.

    Args:
        ticks: Number of ticks to publish. -1 = run indefinitely.
    """
    producer = build_producer()
    count = 0

    try:
        while ticks == -1 or count < ticks:
            event = TickEvent(
                timestamp=int(time.time()),
                token_id=random.choice(MOCK_TOKENS),
                price_usd=round(random.uniform(0.01, 2000.0), 4),
                volume_24h=round(random.uniform(10_000, 5_000_000), 2),
                buy_sell_ratio=round(random.uniform(0.1, 9.9), 2),
            )
            future = producer.send(TOPIC, asdict(event))
            future.get(timeout=10)
            logger.info("Published: %s @ $%.4f", event.token_id, event.price_usd)
            count += 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Stream interrupted by user.")
    except Exception as e:
        logger.error("Unexpected error: %s", e)
    finally:
        producer.flush()
        producer.close()
        logger.info("Producer shut down cleanly.")


if __name__ == "__main__":
    simulate_stream()
