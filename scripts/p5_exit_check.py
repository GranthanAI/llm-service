"""
P5 exit check: a hand-built chat.message.created 2.4 event (billing grant: smart / Auto) -> llm-service answers with a
Smart model -> chat.response.generated 2.2 carries calls (request_analysis + answer with provider-reported tokens) ->
billing records a usage row with a non-zero cost for the request_message_id.

THIS MAKES REAL PROVIDER CALLS (Groq / the Smart-group models), so run it only when you mean to.

    # needs: Kafka reachable at 127.0.0.1:9092 (never `localhost` on Windows: it resolves to IPv6 first),
    #        llm-service running THIS code with GROQ_API_KEY (and MODEL_REGISTRY_FILE=config/models.yaml),
    #        billing API + worker running in simulation (PAYMENT_SIMULATION=true) with the same Kafka.
    uv run python scripts/p5_exit_check.py [--kafka 127.0.0.1:9092] [--user-id <uuid>] [--timeout 90]

It prints the response event's billing fields and `calls`, then the SQL to run against billing's database:
    docker exec -i graphgpt-postgres psql -U postgres -d billing_db -c "<printed query>"
No topic or consumer group is created (the consumer uses no group; nothing is committed).
"""

import argparse
import asyncio
import json
import time
import uuid
from datetime import UTC, datetime

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

IN_TOPIC = "chat.message.created"
OUT_TOPIC = "chat.response.generated"


def build_event(user_id: str, request_message_id: str, response_message_id: str) -> dict:
    return {
        "event_type": "chat.message.created", "schema_version": "2.4",
        "message_id": request_message_id, "response_message_id": response_message_id,
        "conversation_id": str(uuid.uuid4()), "user_id": user_id,
        "content": "Explain in three sentences why Kafka partitions preserve order only within a partition.",
        "mode_hint": "smart",
        "billing": {"hold_id": None, "requested_model": None, "granted_group": "smart", "granted_model": None,
                    "max_context_tokens": 64000, "funded_by": "plan", "downgraded": False, "enforcement": "shadow"},
        "timestamp": datetime.now(UTC).isoformat(),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kafka", default="127.0.0.1:9092")
    ap.add_argument("--user-id", default=str(uuid.uuid4()))
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()
    if "localhost" in args.kafka:
        print("Use 127.0.0.1, not localhost, on Windows.")
        return 2

    request_id, response_id = str(uuid.uuid4()), str(uuid.uuid4())
    consumer = AIOKafkaConsumer(OUT_TOPIC, bootstrap_servers=args.kafka, group_id=None, auto_offset_reset="latest",
                                enable_auto_commit=False)
    await consumer.start()
    producer = AIOKafkaProducer(bootstrap_servers=args.kafka)
    await producer.start()
    try:
        await consumer.getmany(timeout_ms=1000)               # join/assign before sending
        event = build_event(args.user_id, request_id, response_id)
        await producer.send_and_wait(IN_TOPIC, json.dumps(event).encode(), key=event["conversation_id"].encode())
        print(f"sent chat.message.created 2.4  request_message_id={request_id}  user_id={args.user_id}")
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            for batch in (await consumer.getmany(timeout_ms=1000)).values():
                for msg in batch:
                    out = json.loads(msg.value)
                    if out.get("request_message_id") != request_id:
                        continue
                    print(json.dumps({k: out.get(k) for k in (
                        "schema_version", "status", "provider", "hold_id", "model_group", "model", "downgraded",
                        "error_code", "usage")}, indent=2))
                    print("calls:")
                    for c in out.get("calls", []):
                        print("  ", json.dumps(c))
                    purposes = {c["purpose"] for c in out.get("calls", [])}
                    ok = (out.get("schema_version") == "2.2" and out.get("status") == "success"
                          and {"request_analysis", "answer"} <= purposes
                          and all(not c["usage_estimated"] for c in out["calls"]))
                    print("PASS (calls include request_analysis + answer, provider-reported)" if ok else "CHECK THE OUTPUT ABOVE")
                    print("\nbilling database:\n  " + sql(request_id))
                    return 0 if ok else 1
        print("timed out: no chat.response.generated for this request (is llm-service running THIS code?)")
        return 1
    finally:
        await producer.stop()
        await consumer.stop()


def sql(request_id: str) -> str:
    return ("SELECT kind, source_id, model, model_group, input_tokens, cached_input_tokens, output_tokens, aux_input_tokens, "
            "aux_output_tokens, cost_micro_inr, hold_id FROM usage_events "
            f"WHERE source_id = '{request_id}';")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
