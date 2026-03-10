import json
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from csi_vae.messages_queue import MessagesQueue, QueueNotCreatedError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sqs_message(body: dict, receipt_handle: str = "rh-1") -> dict:
    """Create a dict that mimics the structure of an SQS message."""
    return {"Body": json.dumps(body), "ReceiptHandle": receipt_handle}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_boto3(monkeypatch: pytest.MonkeyPatch) -> Generator[MagicMock]:
    """Patch boto3.client so no real AWS calls are made."""
    mock_sqs = MagicMock()
    with patch("boto3.client", return_value=mock_sqs):
        yield mock_sqs


@pytest.fixture
def queue(mock_boto3: MagicMock) -> MessagesQueue:
    """Create a MessagesQueue with a mocked SQS client."""
    mock_boto3.create_queue.return_value = {"QueueUrl": "https://sqs.fake/queue"}
    q = MessagesQueue("us-east-1")
    q.create("test-queue")
    return q


@pytest.fixture
def fresh_queue(mock_boto3: MagicMock) -> MessagesQueue:
    """Provide a MessagesQueue instance that hasn't been created yet."""
    return MessagesQueue("us-east-1")


# ---------------------------------------------------------------------------
# QueueNotCreatedError
# ---------------------------------------------------------------------------


class TestQueueNotCreatedError:
    def test_message(self):

        err = QueueNotCreatedError()
        assert "create()" in str(err)

    def test_is_runtime_error(self):

        assert issubclass(QueueNotCreatedError, RuntimeError)


# ---------------------------------------------------------------------------
# MessagesQueue.from_url
# ---------------------------------------------------------------------------


class TestFromUrl:
    def test_url_is_set(self):

        q = MessagesQueue.from_url("https://sqs.fake/existing", "us-east-1")
        assert q.url == "https://sqs.fake/existing"

    def test_returns_messages_queue_instance(self):

        q = MessagesQueue.from_url("https://sqs.fake/existing", "us-east-1")
        assert isinstance(q, MessagesQueue)


# ---------------------------------------------------------------------------
# MessagesQueue.url property
# ---------------------------------------------------------------------------


class TestUrlProperty:
    def test_raises_before_create(self, fresh_queue):

        with pytest.raises(QueueNotCreatedError):
            _ = fresh_queue.url

    def test_returns_url_after_create(self, queue):
        assert queue.url == "https://sqs.fake/queue"

    def test_raises_after_destroy(self, queue, mock_boto3):

        queue.destroy()
        with pytest.raises(QueueNotCreatedError):
            _ = queue.url


# ---------------------------------------------------------------------------
# MessagesQueue.create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_calls_create_queue(self, fresh_queue, mock_boto3):
        mock_boto3.create_queue.return_value = {"QueueUrl": "https://sqs.fake/new"}
        fresh_queue.create("my-queue")
        mock_boto3.create_queue.assert_called_once_with(QueueName="my-queue")

    def test_sets_url(self, fresh_queue, mock_boto3):
        mock_boto3.create_queue.return_value = {"QueueUrl": "https://sqs.fake/new"}
        fresh_queue.create("my-queue")
        assert fresh_queue.url == "https://sqs.fake/new"


# ---------------------------------------------------------------------------
# MessagesQueue.destroy
# ---------------------------------------------------------------------------


class TestDestroy:
    def test_calls_delete_queue(self, queue, mock_boto3):
        queue.destroy()
        mock_boto3.delete_queue.assert_called_once_with(QueueUrl="https://sqs.fake/queue")

    def test_clears_url(self, queue, mock_boto3):

        queue.destroy()
        with pytest.raises(QueueNotCreatedError):
            _ = queue.url

    def test_noop_when_not_created(self, fresh_queue, mock_boto3):
        fresh_queue.destroy()
        mock_boto3.delete_queue.assert_not_called()


# ---------------------------------------------------------------------------
# MessagesQueue.push
# ---------------------------------------------------------------------------


class TestPush:
    def test_sends_serialised_message(self, queue, mock_boto3):
        payload = {"key": "value"}
        queue.push(payload)
        mock_boto3.send_message.assert_called_once_with(
            QueueUrl="https://sqs.fake/queue",
            MessageBody=json.dumps(payload),
        )

    def test_raises_when_not_created(self, fresh_queue):

        with pytest.raises(QueueNotCreatedError):
            fresh_queue.push({"k": "v"})


# ---------------------------------------------------------------------------
# MessagesQueue.pop
# ---------------------------------------------------------------------------


class TestPop:
    def test_returns_empty_list_when_no_messages(self, queue, mock_boto3):
        mock_boto3.receive_message.return_value = {}
        assert queue.pop() == []

    def test_returns_parsed_messages(self, queue, mock_boto3):
        msg = {"hello": "world"}
        mock_boto3.receive_message.return_value = {"Messages": [make_sqs_message(msg, "rh-1")]}
        result = queue.pop(max_messages=1)
        assert result == [msg]

    def test_deletes_messages_after_receive(self, queue, mock_boto3):
        mock_boto3.receive_message.return_value = {"Messages": [make_sqs_message({"a": 1}, "rh-abc")]}
        queue.pop(max_messages=1)
        mock_boto3.delete_message.assert_called_once_with(
            QueueUrl="https://sqs.fake/queue",
            ReceiptHandle="rh-abc",
        )

    def test_respects_max_messages_cap_per_request(self, queue, mock_boto3):
        """SQS caps at 10 per call; requesting 5 should pass 5."""
        mock_boto3.receive_message.return_value = {}
        queue.pop(max_messages=5)
        mock_boto3.receive_message.assert_called_once_with(
            QueueUrl="https://sqs.fake/queue",
            MaxNumberOfMessages=5,
        )

    def test_loops_for_more_than_10_messages(self, queue, mock_boto3):
        """Requesting 15 messages should trigger two receive_message calls (10 + 5)."""
        batch1 = [make_sqs_message({"i": i}, f"rh-{i}") for i in range(10)]
        batch2 = [make_sqs_message({"i": i + 10}, f"rh-{i + 10}") for i in range(5)]
        mock_boto3.receive_message.side_effect = [
            {"Messages": batch1},
            {"Messages": batch2},
            {},
        ]
        result = queue.pop(max_messages=15)
        assert len(result) == 15
        assert mock_boto3.receive_message.call_count == 2

    def test_stops_early_when_queue_exhausted(self, queue, mock_boto3):
        """If SQS returns fewer messages than requested, stop looping."""
        mock_boto3.receive_message.side_effect = [
            {"Messages": [make_sqs_message({"x": 1}, "rh-1")]},
            {},
        ]
        result = queue.pop(max_messages=20)
        assert len(result) == 1
        assert mock_boto3.receive_message.call_count == 2

    def test_raises_when_not_created(self, fresh_queue):

        with pytest.raises(QueueNotCreatedError):
            fresh_queue.pop()
