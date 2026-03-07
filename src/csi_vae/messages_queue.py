import json

import boto3


class QueueNotCreatedError(RuntimeError):
    """Raised when trying to push to or pop from a queue that hasn't been created yet or is already destroyed."""

    def __init__(self) -> None:
        """Initialize the error with a default message."""
        super().__init__("Queue not created yet or already destroyed. Call create() before using the queue.")


class MessagesQueue:
    """An implementation of MessagesQueue using AWS SQS."""

    def __init__(self) -> None:
        """Initialize the SQS client and store the queue URL."""
        self.__sqs = boto3.client("sqs")
        self._url: str | None = None

    @staticmethod
    def from_url(url: str) -> "MessagesQueue":
        """Create a MessagesQueue instance from an existing SQS queue URL."""
        instance = MessagesQueue()
        instance._url = url
        return instance

    @property
    def url(self) -> str:
        """Return the URL of the SQS queue, or raise an error if it hasn't been created."""
        if self._url is None:
            raise QueueNotCreatedError

        return self._url

    def create(self, name: str) -> None:
        """Create the SQS queue and store its URL."""
        self._url = self.__sqs.create_queue(QueueName=name)["QueueUrl"]

    def destroy(self) -> None:
        """Delete the SQS queue."""
        if self._url is not None:
            self.__sqs.delete_queue(QueueUrl=self._url)
            self._url = None

    def push(self, item: dict) -> None:
        """Push a message to the SQS queue."""
        if self._url is None:
            raise QueueNotCreatedError

        self.__sqs.send_message(QueueUrl=self._url, MessageBody=json.dumps(item))

    def pop(self, max_messages: int = 10) -> list[dict]:
        """Pop messages from the SQS queue, or return an empty list if none are available."""
        if self._url is None:
            raise QueueNotCreatedError

        results = []
        remaining = max_messages

        # SQS caps at 10 per request, so we loop if more are requested
        while remaining > 0:
            resp = self.__sqs.receive_message(
                QueueUrl=self._url,
                MaxNumberOfMessages=min(remaining, 10),
            ).get("Messages", [])
            if not resp:
                break

            for msg in resp:
                results.append(json.loads(msg["Body"]))
                self.__sqs.delete_message(QueueUrl=self._url, ReceiptHandle=msg["ReceiptHandle"])

            remaining -= len(resp)

        return results
