import logging
import traceback
from datetime import datetime
from typing import Any

from csi_vae.messages_queue import MessagesQueue


def _timestamp_to_iso(timestamp: float) -> str:
    """Convert a timestamp to an ISO 8601 formatted string.

    Arguments:
        timestamp: The timestamp to convert, in seconds since the epoch.

    Returns:
        An ISO 8601 formatted string representing the given timestamp.

    """
    return datetime.fromtimestamp(timestamp, tz=datetime.now().astimezone().tzinfo).isoformat()


class StreamHandler(logging.StreamHandler):
    """Logging handler that outputs log records to the console."""

    def __init__(self, study_name: str, trial_number: int, seed: int) -> None:
        """Initialize the StreamHandler."""
        super().__init__()

        self.__study_name = study_name
        self.__trial_number = trial_number
        self.__seed = seed

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by outputting it to the console."""
        message: dict[str, Any] = record.msg if isinstance(record.msg, dict) else {"message": record.getMessage()}

        if record.exc_info:
            message["error"] = str(record.exc_info[1])
            message["traceback"] = traceback.format_exception(*record.exc_info)

        message = {
            **message,
            "date_time": _timestamp_to_iso(record.created),
            "study_name": self.__study_name,
            "trial_number": self.__trial_number,
            "seed": self.__seed,
        }

        super().emit(
            logging.LogRecord(
                name=record.name,
                level=record.levelno,
                pathname=record.pathname,
                lineno=record.lineno,
                msg=message,
                args=(),
                exc_info=record.exc_info,
            ),
        )


class QueueHandler(logging.Handler):
    """Logging handler that sends log records to an AWS SQS queue."""

    def __init__(self, queue: MessagesQueue, study_name: str, trial_number: int, seed: int) -> None:
        """Initialize the QueueHandler with the specified MessagesQueue."""
        super().__init__()
        self.__queue = queue
        self.__study_name = study_name
        self.__trial_number = trial_number
        self.__seed = seed

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record by sending it to the SQS queue."""
        message: dict[str, Any] = record.msg if isinstance(record.msg, dict) else {"message": record.getMessage()}

        if record.exc_info:
            message["error"] = str(record.exc_info[1])
            message["traceback"] = traceback.format_exception(*record.exc_info)

        message = {
            **message,
            "date_time": _timestamp_to_iso(record.created),
            "study_name": self.__study_name,
            "trial_number": self.__trial_number,
            "seed": self.__seed,
        }
        try:
            self.__queue.push(message)
        except Exception:  # noqa: BLE001
            self.handleError(record)  # Silently fails to avoid log loops
