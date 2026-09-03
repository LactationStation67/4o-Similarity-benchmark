#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
OpenRouter Client – Simple Async Wrapper
================================================================================

This client allows you to send chat completion requests to any model
available on OpenRouter (https://openrouter.ai).

Usage:
    from openrouter_client import OpenRouterClient

    client = OpenRouterClient(api_key="your-openrouter-api-key")
    response = await client.generate(
        model="openai/gpt-4o-2024-11-20",
        system_prompt="You are a helpful assistant.",
        user_prompt="Hello, how are you?",
        temperature=0.7
    )
    print(response)
    await client.close()

================================================================================
"""

import json
import httpx
from typing import Optional, List, Dict, Any, Union


class OpenRouterClient:
    """
    Async client for OpenRouter's chat completions API.
    """

    def __init__(
        self,
        api_key: str,   # 👈 REQUIRED – get your key from https://openrouter.ai/keys
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 120.0,
    ):
        """
        Args:
            api_key: Your OpenRouter API key (starts with "sk-or-v1-...").
            base_url: The OpenRouter API endpoint (defaults to v1).
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

        # HTTP client with persistent connection pooling
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=30,
            ),
        )

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # =========================================================================
    #  Chat Completion
    # =========================================================================

    async def generate(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Send a chat completion request.

        Args:
            model: The full model ID (e.g. "openai/gpt-4o-2024-11-20").
            system_prompt: System instruction (optional).
            user_prompt: User message (optional).
            messages: Full message list (overrides system+user if given).
            temperature: Sampling temperature (0.0–1.0).
            max_tokens: Maximum tokens to generate.
            stream: If True, returns an async generator.
            **kwargs: Additional parameters for the API (e.g. `top_p`, `frequency_penalty`).

        Returns:
            - If stream=False: the assistant's text (or error dict on failure).
            - If stream=True: an async generator yielding tokens.

        Raises:
            httpx.HTTPError: On network errors.
        """
        # Build messages
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})

        # Prepare payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }

        json_body = json.dumps(payload).encode("utf-8")

        try:
            if stream:
                async def stream_generator():
                    async with self.client.stream(
                        "POST",
                        "/chat/completions",
                        content=json_body,
                        headers=self.headers,
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line and line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data)
                                    if chunk.get("choices"):
                                        delta = chunk["choices"][0].get("delta", {})
                                        if delta.get("content"):
                                            yield delta["content"]
                                except json.JSONDecodeError:
                                    pass
                return stream_generator()

            # Non‑streaming
            resp = await self.client.post(
                "/chat/completions",
                content=json_body,
                headers=self.headers,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("choices"):
                return data["choices"][0]["message"]["content"]
            return data

        except httpx.HTTPError as e:
            return {"error": str(e), "status": "request_failed"}

    # =========================================================================
    #  Sync Wrapper
    # =========================================================================

    def generate_sync(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """
        Synchronous wrapper for `generate()` – useful for scripts that aren't async.
        """
        import asyncio

        # Check if we're already in an async loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return asyncio.run(
                self.generate(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    **kwargs,
                )
            )
        else:
            # We're inside an async environment – run in a separate thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.generate(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                        **kwargs,
                    ),
                )
                return future.result()

    # =========================================================================
    #  Cleanup
    # =========================================================================

    async def close(self):
        """Close the underlying HTTP client."""
        await self.client.aclose()

    def __del__(self):
        """Try to close the client on garbage collection."""
        try:
            import asyncio

            if self.client._transport is not None:
                asyncio.create_task(self.close())
        except Exception:
            pass


# =========================================================================
#  Convenience One‑Shot Function
# =========================================================================

async def generate(
    prompt: str,
    system: str = "You are a helpful AI assistant.",
    model: str = "openai/gpt-4o-2024-11-20",
    api_key: str = "",  # 👈 Paste your OpenRouter key here
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """
    Quick one-shot generation without managing a client instance.
    """
    client = OpenRouterClient(api_key=api_key)
    try:
        response = await client.generate(
            model=model,
            system_prompt=system,
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response
    finally:
        await client.close()


# =========================================================================
#  Example Usage
# =========================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        # Replace with your actual key
        API_KEY = "sk-or-v1-..."  # <-- INSERT YOUR OPENROUTER KEY

        client = OpenRouterClient(api_key=API_KEY)

        # Simple chat
        response = await client.generate(
            model="openai/gpt-4o-2024-11-20",
            system_prompt="You are a helpful assistant.",
            user_prompt="What is the capital of France?",
            temperature=0.7,
        )
        print("Response:", response)

        # Multi‑turn conversation
        messages = [
            {"role": "system", "content": "You are a grumpy old man."},
            {"role": "user", "content": "How are you today?"},
        ]
        response = await client.generate(
            model="openai/gpt-4o-2024-11-20",
            messages=messages,
            temperature=0.5,
        )
        print("Grumpy response:", response)

        await client.close()

    asyncio.run(main())
