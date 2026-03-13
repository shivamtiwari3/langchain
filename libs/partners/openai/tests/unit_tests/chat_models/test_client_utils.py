"""Unit tests for _client_utils async httpx client helpers."""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_openai.chat_models._client_utils import (
    _AsyncHttpxClientWrapper,
    _get_default_async_httpx_client,
    _get_loop_aware_async_httpx_client,
    _LoopAwareAsyncHttpxClientWrapper,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_in_new_loop(coro: Any) -> Any:
    """Run a coroutine in a brand-new event loop and return the result."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# _LoopAwareAsyncHttpxClientWrapper — inner-client isolation
# ---------------------------------------------------------------------------


def test_same_loop_reuses_inner_client() -> None:
    """Two sends from the same loop must share one inner client."""
    proxy = _LoopAwareAsyncHttpxClientWrapper(None, 60.0)

    async def get_two_clients() -> (
        tuple[_AsyncHttpxClientWrapper, _AsyncHttpxClientWrapper]
    ):
        c1 = proxy._get_client_for_current_loop()
        c2 = proxy._get_client_for_current_loop()
        return c1, c2

    c1, c2 = _run_in_new_loop(get_two_clients())
    assert c1 is c2


def test_different_loops_get_different_inner_clients() -> None:
    """Each event loop must receive its own inner httpx client."""
    proxy = _LoopAwareAsyncHttpxClientWrapper(None, 60.0)
    clients: list[_AsyncHttpxClientWrapper] = []

    async def capture_client() -> None:
        clients.append(proxy._get_client_for_current_loop())

    _run_in_new_loop(capture_client())
    _run_in_new_loop(capture_client())

    assert len(clients) == 2
    assert clients[0] is not clients[1]


def test_different_loops_in_threads_get_different_inner_clients() -> None:
    """Loops running in separate threads must each get an isolated inner client."""
    proxy = _LoopAwareAsyncHttpxClientWrapper(None, 60.0)
    clients: list[_AsyncHttpxClientWrapper] = []
    lock = threading.Lock()

    def thread_body() -> None:
        async def capture() -> None:
            with lock:
                clients.append(proxy._get_client_for_current_loop())

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(capture())
        finally:
            loop.close()

    t1 = threading.Thread(target=thread_body)
    t2 = threading.Thread(target=thread_body)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(clients) == 2
    assert clients[0] is not clients[1]


def test_loop_gc_removes_inner_client_entry() -> None:
    """After a loop is GC'd the WeakKeyDictionary entry should be gone."""
    import gc
    import weakref

    proxy = _LoopAwareAsyncHttpxClientWrapper(None, 60.0)

    async def populate() -> None:
        proxy._get_client_for_current_loop()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(populate())
    assert len(proxy._loop_clients) == 1

    loop.close()
    loop_ref = weakref.ref(loop)
    del loop
    gc.collect()

    if loop_ref() is None:
        # Loop was GC'd; WeakKeyDictionary should be empty.
        assert len(proxy._loop_clients) == 0


def test_send_delegates_to_inner_client() -> None:
    """send() must call the inner client's send, not the proxy's own transport."""
    proxy = _LoopAwareAsyncHttpxClientWrapper(None, 60.0)

    fake_response = MagicMock()
    fake_inner = AsyncMock()
    fake_inner.send = AsyncMock(return_value=fake_response)

    async def run() -> Any:
        with patch.object(
            proxy, "_get_client_for_current_loop", return_value=fake_inner
        ):
            import httpx

            req = httpx.Request("GET", "https://example.com")
            return await proxy.send(req)

    result = _run_in_new_loop(run())
    assert result is fake_response
    fake_inner.send.assert_awaited_once()


def test_aclose_closes_current_loop_inner_client() -> None:
    """aclose() must close only the inner client for the calling loop."""
    proxy = _LoopAwareAsyncHttpxClientWrapper(None, 60.0)

    async def run() -> None:
        # Populate the inner client for this loop.
        inner = proxy._get_client_for_current_loop()
        assert len(proxy._loop_clients) == 1
        with patch.object(inner, "aclose", new_callable=AsyncMock) as mock_close:
            await proxy.aclose()
            mock_close.assert_awaited_once()
        # Entry removed after aclose.
        assert len(proxy._loop_clients) == 0

    _run_in_new_loop(run())


# ---------------------------------------------------------------------------
# _get_loop_aware_async_httpx_client — lru_cache behaviour
# ---------------------------------------------------------------------------


def test_same_params_return_same_proxy() -> None:
    """_get_loop_aware_async_httpx_client must be @lru_cache'd."""
    p1 = _get_loop_aware_async_httpx_client(None, 60.0)
    p2 = _get_loop_aware_async_httpx_client(None, 60.0)
    assert p1 is p2


def test_different_params_return_different_proxies() -> None:
    p1 = _get_loop_aware_async_httpx_client(None, 60.0)
    p2 = _get_loop_aware_async_httpx_client(None, 30.0)
    assert p1 is not p2


# ---------------------------------------------------------------------------
# _get_default_async_httpx_client — routing logic
# ---------------------------------------------------------------------------


def test_hashable_timeout_returns_loop_aware_proxy() -> None:
    client = _get_default_async_httpx_client(None, 60.0)
    assert isinstance(client, _LoopAwareAsyncHttpxClientWrapper)


def test_unhashable_timeout_returns_plain_wrapper() -> None:
    """An unhashable timeout (e.g. httpx.Timeout) bypasses caching."""
    import httpx

    timeout = httpx.Timeout(10.0)
    client = _get_default_async_httpx_client(None, timeout)
    # Must be the plain wrapper, not the loop-aware proxy.
    assert isinstance(client, _AsyncHttpxClientWrapper)
    assert not isinstance(client, _LoopAwareAsyncHttpxClientWrapper)


def test_same_hashable_params_return_same_proxy_instance() -> None:
    """Repeated calls with the same args share the cached proxy."""
    c1 = _get_default_async_httpx_client(None, 60.0)
    c2 = _get_default_async_httpx_client(None, 60.0)
    assert c1 is c2
