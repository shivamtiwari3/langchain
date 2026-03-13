"""Helpers for creating OpenAI API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatOpenAI.

Logic is largely replicated from openai._base_client.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import weakref
from collections.abc import Awaitable, Callable
from functools import lru_cache
from threading import Lock
from typing import Any, cast

import httpx
import openai
from pydantic import SecretStr


class _SyncHttpxClientWrapper(openai.DefaultHttpxClient):
    """Borrowed from openai._base_client."""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncHttpxClientWrapper(openai.DefaultAsyncHttpxClient):
    """Borrowed from openai._base_client."""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass


def _build_sync_httpx_client(
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    return _SyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )


def _build_async_httpx_client(
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    return _AsyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )


@lru_cache
def _cached_sync_httpx_client(
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    return _build_sync_httpx_client(base_url, timeout)


@lru_cache
def _cached_async_httpx_client(
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    return _build_async_httpx_client(base_url, timeout)


class _LoopAwareAsyncHttpxClientWrapper(openai.DefaultAsyncHttpxClient):
    """Proxy that dispatches async requests to a per-event-loop inner httpx client.

    The outer proxy is cached via `@lru_cache` (one instance per `(base_url, timeout)`
    pair), preserving `ChatOpenAI` instantiation performance.

    Each event loop gets its own inner `_AsyncHttpxClientWrapper`. This prevents
    `APIConnectionError` caused by reusing `asyncio.Lock`-bound connection pools
    across different event loops — the bug that occurs when `@lru_cache` returns a
    single shared `httpx.AsyncClient` for all loops.

    When an event loop is garbage-collected, `WeakKeyDictionary` automatically
    removes the corresponding inner client entry.
    """

    def __init__(self, base_url: str | None, timeout: Any) -> None:
        effective_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        )
        super().__init__(base_url=effective_url, timeout=timeout)
        self._lc_base_url = base_url
        self._lc_timeout = timeout
        self._loop_clients: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, _AsyncHttpxClientWrapper
        ] = weakref.WeakKeyDictionary()
        self._lock = Lock()

    def _get_client_for_current_loop(self) -> _AsyncHttpxClientWrapper:
        """Get or create an inner httpx client bound to the current event loop."""
        loop = asyncio.get_running_loop()
        with self._lock:
            if loop not in self._loop_clients:
                self._loop_clients[loop] = _build_async_httpx_client(
                    self._lc_base_url, self._lc_timeout
                )
        return self._loop_clients[loop]

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        """Send request via the inner client bound to the current event loop."""
        return await self._get_client_for_current_loop().send(request, **kwargs)

    async def aclose(self) -> None:
        """Close the inner client for the current event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        with self._lock:
            client = self._loop_clients.pop(loop, None)
        if client is not None:
            await client.aclose()


@lru_cache
def _get_loop_aware_async_httpx_client(
    base_url: str | None, timeout: Any
) -> _LoopAwareAsyncHttpxClientWrapper:
    return _LoopAwareAsyncHttpxClientWrapper(base_url, timeout)


def _get_default_httpx_client(
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    """Get default httpx client.

    Uses cached client unless timeout is `httpx.Timeout`, which is not hashable.
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_sync_httpx_client(base_url, timeout)
    else:
        return _cached_sync_httpx_client(base_url, timeout)


def _get_default_async_httpx_client(
    base_url: str | None, timeout: Any
) -> _LoopAwareAsyncHttpxClientWrapper | _AsyncHttpxClientWrapper:
    """Get default async httpx client.

    Returns a `_LoopAwareAsyncHttpxClientWrapper` proxy when the timeout is
    hashable (cached path). The proxy is shared across all `ChatOpenAI` instances
    with the same `(base_url, timeout)` but internally routes each request to an
    inner `httpx.AsyncClient` bound to the calling event loop, preventing
    `APIConnectionError` from cross-loop connection-pool reuse.

    Falls back to a plain `_AsyncHttpxClientWrapper` when timeout is not hashable
    (e.g., an `httpx.Timeout` object), since `@lru_cache` requires hashable keys.
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_async_httpx_client(base_url, timeout)
    else:
        return _get_loop_aware_async_httpx_client(base_url, timeout)


def _resolve_sync_and_async_api_keys(
    api_key: SecretStr | Callable[[], str] | Callable[[], Awaitable[str]],
) -> tuple[str | None | Callable[[], str], str | Callable[[], Awaitable[str]]]:
    """Resolve sync and async API key values.

    Because OpenAI and AsyncOpenAI clients support either sync or async callables for
    the API key, we need to resolve separate values here.
    """
    if isinstance(api_key, SecretStr):
        sync_api_key_value: str | None | Callable[[], str] = api_key.get_secret_value()
        async_api_key_value: str | Callable[[], Awaitable[str]] = (
            api_key.get_secret_value()
        )
    elif callable(api_key):
        if inspect.iscoroutinefunction(api_key):
            async_api_key_value = api_key
            sync_api_key_value = None
        else:
            sync_api_key_value = cast(Callable, api_key)

            async def async_api_key_wrapper() -> str:
                return await asyncio.get_event_loop().run_in_executor(
                    None, cast(Callable, api_key)
                )

            async_api_key_value = async_api_key_wrapper

    return sync_api_key_value, async_api_key_value
