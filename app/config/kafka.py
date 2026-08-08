"""
Kafka configuration models and utilities.
"""

from pydantic import BaseModel, SecretStr


class KafkaConfig(BaseModel):
    bootstrap_servers: str
    consumer_group: str
    input_topic: str
    output_topic: str
    chunk_topic: str
    dlq_topic: str
    max_poll_interval_ms: int = 300000
    username: str | None = None
    password: SecretStr | None = None
