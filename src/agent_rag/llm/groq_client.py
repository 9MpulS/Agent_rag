"""Async Groq client with token and latency tracking."""

import json
import time
from dataclasses import dataclass
import structlog
from groq import AsyncGroq
from agent_rag.config import settings

logger = structlog.get_logger()

@dataclass
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float

@dataclass
class LLMResponse:
    content: str
    usage: LLMUsage

class GroqClient:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * settings.GROQ_PRICE_INPUT_PER_TOKEN +
            completion_tokens * settings.GROQ_PRICE_OUTPUT_PER_TOKEN
        )

    async def complete(self, system: str, user: str) -> LLMResponse:
        """Standard completion call."""
        t0 = time.perf_counter()
        
        response = await self.client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
        )
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        usage = LLMUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency_ms=latency_ms,
            cost_usd=self._calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
        )
        
        return LLMResponse(content=response.choices[0].message.content, usage=usage)

    async def complete_json(self, system: str, user: str) -> tuple[dict, LLMUsage]:
        """Completion call forcing JSON output."""
        t0 = time.perf_counter()
        
        response = await self.client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        usage = LLMUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency_ms=latency_ms,
            cost_usd=self._calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
        )
        
        content = response.choices[0].message.content
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.error("groq_json_parse_error", content=content)
            parsed = {}
            
        return parsed, usage

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[object, LLMUsage]:
        """Completion call with native tool calling support.

        Args:
            messages: Full conversation history in OpenAI format.
            tools: Tool definitions in OpenAI function calling format.

        Returns:
            Tuple of (response message object, usage).
            The message may contain tool_calls or text content.
        """
        t0 = time.perf_counter()

        response = await self.client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            tools=tools,
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        usage = LLMUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency_ms=latency_ms,
            cost_usd=self._calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            ),
        )

        return response.choices[0].message, usage

