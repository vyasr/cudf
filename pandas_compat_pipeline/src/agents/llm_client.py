# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async LLM client for NVIDIA Inference Hub models via litellm.

Provides:
- LLMClient class for calling Claude 4.6 Opus (fixer) and GPT 5.5 (reviewer)
- Automatic retry with exponential backoff
- Per-call timeouts
- Rate limiting (max 4 concurrent calls)
"""

import asyncio
import json
import logging
import os
from typing import Any, cast

import litellm
from litellm import ModelResponse

logger = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client for NVIDIA Inference Hub models."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://inference-api.nvidia.com/v1",
        max_concurrent: int = 4,
    ):
        """Initialize LLM client.

        Args:
            api_key: INFERENCE_HUB_KEY. If None, reads from environment.
            base_url: NVIDIA Inference Hub endpoint
            max_concurrent: Max concurrent LLM calls (rate limiting)
        """
        self.api_key = api_key or os.environ.get("INFERENCE_HUB_KEY")
        if not self.api_key:
            raise ValueError(
                "INFERENCE_HUB_KEY not found in environment and not provided"
            )
        self.base_url = base_url
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def call_fixer(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Call Claude 4.6 Opus for test fixing with tool support.

        Args:
            messages: Chat messages for Claude
            tools: Optional tool definitions for function calling

        Returns:
            Model response text

        Raises:
            Exception: If all retries fail
        """
        model = "openai/aws/anthropic/bedrock-claude-opus-4-6"
        timeout = 300  # 5 minutes
        return await self._call_with_retry(
            model=model,
            messages=messages,
            tools=tools,
            timeout=timeout,
            max_retries=3,
        )

    async def call_reviewer(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Call GPT 5.5 for test review.

        Args:
            messages: Chat messages for GPT

        Returns:
            Model response text

        Raises:
            Exception: If all retries fail
        """
        model = "openai/openai/openai/gpt-5.5"  # OpenAI model via NVIDIA hub
        timeout = 180  # 3 minutes
        return await self._call_with_retry(
            model=model,
            messages=messages,
            tools=None,
            timeout=timeout,
            max_retries=3,
        )

    async def _call_with_retry(
        self,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        timeout: int = 300,
        max_retries: int = 3,
    ) -> str:
        """Call LLM with exponential backoff retry.

        Args:
            model: Model identifier
            messages: Chat messages
            tools: Optional tool definitions
            timeout: Call timeout in seconds
            max_retries: Number of retries (3 = up to 4 attempts total)

        Returns:
            Model response text

        Raises:
            Exception: If all retries fail
        """
        base_delay = 2  # seconds
        max_delay = 30  # seconds
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                async with self._semaphore:
                    response = await asyncio.wait_for(
                        litellm.acompletion(
                            model=model,
                            messages=messages,
                            api_key=self.api_key,
                            api_base=self.base_url,
                            tools=tools,
                            timeout=timeout,
                        ),
                        timeout=timeout
                        + 10,  # add buffer for internal timeouts
                    )

                # acompletion returns ModelResponse when stream=False (our case).
                # The return type is typed as ModelResponse | CustomStreamWrapper,
                # so we narrow with cast since we never pass stream=True.
                result = cast(ModelResponse, response)

                # Extract message content
                if result.choices:
                    message = result.choices[0].message
                    content = message.content
                    if content:
                        return str(content)
                    # Handle tool calls if present
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        return json.dumps(
                            [tc.model_dump() for tc in message.tool_calls]
                        )

                raise ValueError("Empty response from model")

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    f"Timeout on {model} attempt {attempt + 1}/{max_retries + 1}"
                )
            except (
                litellm.APIConnectionError,
                litellm.RateLimitError,
                litellm.APIError,
            ) as e:
                last_error = e
                logger.warning(
                    f"API error on {model} attempt {attempt + 1}/{max_retries + 1}: {e}"
                )

            # Backoff before retry (but not after last attempt)
            if attempt < max_retries:
                delay = min(base_delay * (2**attempt), max_delay)
                logger.info(f"Retrying after {delay}s delay...")
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Failed to call {model} after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )
