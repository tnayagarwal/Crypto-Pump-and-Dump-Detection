"""
Kafka Consumer - Real-Time Anomaly Stream Processor
====================================================
Subscribes to the crypto_ticks Kafka topic and routes incoming tick events
to the anomaly detection model for real-time pump-and-dump classification.
"""

import json
import logging
import time
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "crypto_ticks"
MAX_POLL_RECORDS = 500  # Batch processing cap for memory safety
GROUP_ID = "anomaly-detector-group"


def build_consumer(max_retries: int = 5) -> KafkaConsumer:
    """Create a KafkaConsumer with connection retry backoff."""
    for attempt in range(max_retries):
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=[KAFKA_BOOTSTRAP],
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id=GROUP_ID,
                max_poll_records=MAX_POLL_RECORDS,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            )
            logger.info("Kafka consumer subscribed to topic: %s", TOPIC)
            return consumer
        except NoBrokersAvailable:
            logger.warning("Broker unavailable. Retry %d/%d...", attempt + 1, max_retries)
            time.sleep(2 ** attempt)
    raise RuntimeError("Could not connect to Kafka broker after retries.")


def process_tick(tick: dict) -> None:
    """
    Route a single tick event to the detection pipeline.
    Currently logs the event; wire to AnomalyDetector.forward() for live inference.
    """
    token = tick.get("token_id", "UNKNOWN")
    price = tick.get("price_usd", 0.0)
    bsr = tick.get("buy_sell_ratio", 1.0)

    # Heuristic pre-filter: flag suspiciously high buy/sell ratio
    if bsr > 8.0:
        logger.warning("SUSPICIOUS: %s - BSR=%.2f @ $%.4f", token, bsr, price)
    else:
        logger.info("Normal tick: %s @ $%.4f", token, price)


def consume_stream() -> None:
    """Main consumer loop - processes ticks indefinitely until interrupted."""
    consumer = build_consumer()
    try:
        for message in consumer:
            tick = message.value
            process_tick(tick)
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    except Exception as e:
        logger.error("Stream processing error: %s", e)
    finally:
        consumer.close()
        logger.info("Consumer shut down cleanly.")


if __name__ == "__main__":
    consume_stream()
