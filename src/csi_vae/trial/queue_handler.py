import logging
import traceback
from typing import Any

from csi_vae.messages_queue import MessagesQueue


class QueueHandler(logging.Handler):
    """Logging handler that sends log records to an AWS SQS queue."""

    def __init__(self, queue: MessagesQueue) -> None:
        """Initialize the QueueHandler with the specified MessagesQueue."""
        super().__init__()
        self.__queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by sending it to the SQS queue."""
        message: dict[str, Any] = record.msg if isinstance(record.msg, dict) else {"message": record.getMessage()}

        if record.exc_info:
            message["error"] = str(record.exc_info[1])
            message["traceback"] = traceback.format_exception(*record.exc_info)

        try:
            self.__queue.push(
                {"timestamp": record.created, **message},
            )
        except Exception:  # noqa: BLE001
            self.handleError(record)  # Silently fails to avoid log loops
