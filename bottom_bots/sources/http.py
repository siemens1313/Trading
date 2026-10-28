"""Tiny stdlib HTTP client: proxy-aware (urllib reads HTTPS_PROXY), retrying, typed errors."""

from __future__ import annotations

import time
import urllib.error
import urllib.request

# Some of these endpoints refuse Python's default user agent.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

_NO_RETRY = {400, 401, 403, 404, 410, 418, 451}


class FetchError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status  # HTTP status, or None when the host couldn't be reached at all


def get(url: str, *, timeout: float = 20.0, retries: int = 2) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in _NO_RETRY:
                raise FetchError(f"HTTP {exc.code} from {url}", exc.code) from exc
            error = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if "Tunnel connection failed" in str(exc):  # a proxy refusing the host won't change its mind
                raise FetchError(f"{url}: {exc}") from exc
            error = exc
        if attempt < retries:
            time.sleep(2**attempt)
    raise FetchError(f"{url}: {error}")
