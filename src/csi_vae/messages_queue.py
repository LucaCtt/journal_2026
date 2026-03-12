import json
from enum import StrEnum

import boto3


class MessageType(StrEnum):
    """Enumeration of possible trial statuses."""

    STARTING = "STARTING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class QueueNotCreatedError(RuntimeError):
    """Raised when trying to push to or pop from a queue that hasn't been created yet or is already destroyed."""

    def __init__(self) -> None:
        """Initialize the error with a default message."""
        super().__init__("Queue not created yet or already destroyed. Call create() before using the queue.")


class MessagesQueue:
    """An implementation of MessagesQueue using AWS SQS."""

    def __init__(self, region_name: str) -> None:
        """Initialize the SQS client and store the queue URL."""
        self.__sqs = boto3.client("sqs", region_name=region_name)
        self.__region_name = region_name
        self.__url: str | None = None

    @staticmethod
    def from_url(url: str, region_name: str) -> "MessagesQueue":
        """Create a MessagesQueue instance from an existing SQS queue URL."""
        instance = MessagesQueue(region_name)
        instance.__url = url
        return instance

    @property
    def url(self) -> str:
        """Return the URL of the SQS queue, or raise an error if it hasn't been created."""
        if self.__url is None:
            raise QueueNotCreatedError

        return self.__url

    @property
    def region_name(self) -> str:
        """Return the AWS region of the SQS client."""
        return self.__region_name

    def create(self, name: str) -> None:
        """Create the SQS queue and store its URL."""
        self.__url = self.__sqs.create_queue(QueueName=name)["QueueUrl"]

    def destroy(self) -> None:
        """Delete the SQS queue."""
        if self.__url is not None:
            self.__sqs.delete_queue(QueueUrl=self.__url)
            self.__url = None

    def push(self, item: dict) -> None:
        """Push a message to the SQS queue."""
        if self.__url is None:
            raise QueueNotCreatedError

        self.__sqs.send_message(QueueUrl=self.__url, MessageBody=json.dumps(item))

    def pop(self, max_messages: int = 10) -> list[dict]:
        """Pop messages from the SQS queue, or return an empty list if none are available."""
        if self.__url is None:
            raise QueueNotCreatedError

        results = []
        remaining = max_messages

        # SQS caps at 10 per request, so we loop if more are requested
        while remaining > 0:
            resp = self.__sqs.receive_message(
                QueueUrl=self.__url,
                MaxNumberOfMessages=min(remaining, 10),
            ).get("Messages", [])
            if not resp:
                break

            for msg in resp:
                results.append(json.loads(msg["Body"]))
                self.__sqs.delete_message(QueueUrl=self.__url, ReceiptHandle=msg["ReceiptHandle"])

            remaining -= len(resp)

        return results
