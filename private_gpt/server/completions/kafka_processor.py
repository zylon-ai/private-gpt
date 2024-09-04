# kafka_processor.py
from confluent_kafka import Consumer, Producer
from pydantic import BaseModel, ValidationError
from typing import Optional, List
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
)
from private_gpt.server.chat.chat_router import ChatBody, chat_completion

import json

# Kafka configuration variables
KAFKA_ADDRESS = 'localhost'
KAFKA_PORT = 9002

class CompletionsBody(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "Always format your response as a valid JSON object, even if the request doesn't explicitly ask for it."
    use_context: bool = False
    context_filter: Optional[ContextFilter] = None
    include_sources: bool = True
    stream: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "How do you fry an egg?",
                    "system_prompt": "You are a rapper. Always answer with a rap.",
                    "stream": False,
                    "use_context": False,
                    "include_sources": False,
                }
            ]
        }
    }


def convert_body_to_messages(body: CompletionsBody) -> List[OpenAIMessage]:
    messages = [OpenAIMessage(content=body.prompt, role="user")]
    if body.system_prompt:
        messages.insert(0, OpenAIMessage(content=body.system_prompt, role="system"))
    return messages


def process_message(message_value: str) -> str:  # Return type is now str (JSON string)
    try:
        body = CompletionsBody.parse_raw(message_value)
        chat_body = ChatBody(
            messages=convert_body_to_messages(body),
            use_context=body.use_context,
            stream=body.stream,
            include_sources=body.include_sources,
            context_filter=body.context_filter,

        )
        completion_response = chat_completion(request=None, chat_body=chat_body)
        # Wrap the successful response in a JSON structure with status
        return json.dumps({
            "status": "success",
            "data": completion_response.model_dump_json()  # Assuming model_dump_json returns a dict
        })
    except ValidationError as e:
        # Return a JSON structure with error details and status
        return json.dumps({
            "status": "error",
            "exception": str(e),
            "location": "process_message - Parsing input message"
        })

def consume_messages(consumer: Consumer, producer: Producer, output_topic: str):
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
        print(f"Received message: {msg.value().decode('utf-8')}")


        completion_response = process_message(msg.value().decode('utf-8'))
        producer.produce(output_topic, completion_response)  # Send the JSON string directly
        producer.flush()

        consumer.commit(asynchronous=False)


def main():
    bootstrap_servers = f"{KAFKA_ADDRESS}:{KAFKA_PORT}"

    consumer_config = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': 'completions-group',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False
    }

    producer_config = {
        'bootstrap.servers': bootstrap_servers
    }

    input_topic = 'prompt_request'  # Updated input topic
    output_topic = 'prompt_response'  # Updated output topic

    consumer = Consumer(consumer_config)
    producer = Producer(producer_config)

    consumer.subscribe([input_topic])

    try:
        consume_messages(consumer, producer, output_topic)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    main()