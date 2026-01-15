# Crypto Pump & Dump Detection

A real-time streaming pipeline that detects coordinated cryptocurrency pump-and-dump
schemes using Apache Kafka for data ingestion and a dual LSTM + GNN model for
temporal and structural anomaly classification.

## Architecture

```
Kafka Producer (Mock Tick Stream)
        |
        v
Kafka Topic: crypto_ticks
        |
        v
Kafka Consumer (Batch-Capped, Retry-Enabled)
        |
        v
AnomalyDetector (LSTM Temporal + GCN Structural)
        |
        v
Classification: Normal | Pump & Dump
```

## Stack
- **Streaming:** Apache Kafka + Zookeeper (Dockerized)
- **Models:** PyTorch 2.1, PyTorch-Geometric (GCN + LSTM)
- **Pipeline:** Python dataclasses, scikit-learn metrics
- **Testing:** Pytest with shape/probability/null-graph coverage

## Running Locally

```bash
# 1. Start Kafka infrastructure
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start producer (publishes mock ticks)
python -m src.data.kafka_producer

# 4. Start consumer (classifies in real time)
python -m src.data.kafka_consumer

# 5. Run tests
pytest tests/ -v
```

> **Note:** Real market datasets and trained weights are not included.
