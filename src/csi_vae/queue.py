import json
import queue
from typing import Protocol

from botocore.client import BaseClient


class TrialMessagesQueue(Protocol):
    """A queue for managing trial messages."""

    url: str

    def push(self, item: dict) -> None:
        """Push a message to the queue.

        Arguments:
            item: The message to push, typically containing trial information and results.

        """
        ...

    def pop(self, max_messages: int = 10) -> list[dict]:
        """Pop a message from the queue, or return None if empty.

        Arguments:
            max_messages: Maximum number of messages to pop at once (for batch processing).

        """
        ...


class SQSTrialMessagesQueue:
    """An implementation of TrialMessagesQueue using AWS SQS."""

    def __init__(self, sqs: BaseClient, queue_url: str) -> None:
        """Initialize the SQS client and store the queue URL."""
        self.__sqs = sqs
        self.url = queue_url

    def push(self, item: dict) -> None:
        """Push a message to the SQS queue."""
        self.__sqs.send_message(QueueUrl=self.url, MessageBody=json.dumps(item))

    def pop(self, max_messages: int = 10) -> list[dict]:
        """Pop messages from the SQS queue, or return an empty list if none are available."""
        results = []
        remaining = max_messages

        # SQS caps at 10 per request, so we loop if more are requested
        while remaining > 0:
            resp = self.__sqs.receive_message(
                QueueUrl=self.url,
                MaxNumberOfMessages=min(remaining, 10),
            ).get("Messages", [])
            if not resp:
                break

            for msg in resp:
                results.append(json.loads(msg.get("Body", "{}")))
                if "ReceiptHandle" in msg:
                    self.__sqs.delete_message(QueueUrl=self.url, ReceiptHandle=msg["ReceiptHandle"])

            remaining -= len(resp)

        return results


class LocalTrialMessagesQueue:
    """An in-memory implementation of TrialMessagesQueue for local testing."""

    def __init__(self) -> None:
        """Initialize the in-memory queue."""
        self.__queue: queue.Queue[dict] = queue.Queue()

    def push(self, item: dict) -> None:
        """Push a message to the in-memory queue."""
        self.__queue.put(item)

    def pop(self, n_messages: int = 1) -> list[dict]:
        """Pop messages from the in-memory queue, or return an empty list if none are available."""
        results = []

        for _ in range(n_messages):
            try:
                results.append(self.__queue.get_nowait())
            except queue.Empty:
                break

        return results
