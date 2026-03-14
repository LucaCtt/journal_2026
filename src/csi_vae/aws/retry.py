import logging
from collections.abc import Callable

from botocore.exceptions import ClientError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_RETRYABLE_ERROR_CODES = frozenset(
    {
        "RequestThrottled",
        "ThrottlingException",
        "ServiceUnavailable",
        "InternalError",
        "SlowDown",
    },
)


def _is_retryable(exc: BaseException) -> bool:
    """Determine if the exception is retryable based on its type and error code.

    Arguments:
        exc: The exception to check.

    Returns:
        True if the exception is retryable, False otherwise.

    """
    return isinstance(exc, ClientError) and exc.response["Error"]["Code"] in _RETRYABLE_ERROR_CODES


def aws_retry(func: Callable) -> Callable:
    """Apply retry logic to AWS operations using tenacity."""
    return retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
    )(func)
