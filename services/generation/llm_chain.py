"""
llm_chain.py — Stage 10: LLM Generation (Gemini via OpenAI-compatible SDK)
===========================================================================
WHY USE THE OPENAI SDK FOR GEMINI?
  Google provides an OpenAI-compatible REST endpoint:
    https://generativelanguage.googleapis.com/v1beta/openai/

  This means we can use the battle-tested langchain-openai / openai SDK
  and simply point it at Google's URL with our Google API key.

  BENEFITS:
  - No langchain-google-genai version conflicts
  - No v1beta/v1 API confusion
  - OpenAI SDK is extremely well-tested and stable
  - Full streaming support out of the box
  - Same code works if we ever switch back to OpenAI

MODEL: gemini-2.0-flash-lite
  - Lightweight and extremely fast
  - Free tier: generous rate limits
  - Strong instruction-following for citation-grounded RAG
  - Context window: 1M tokens

WHY LOW TEMPERATURE (0.1)?
  We want factual extraction, not creative generation.
  Low temperature keeps the model deterministic and grounded.

STREAMING:
  Tokens arrive in ~200ms vs waiting 5+ seconds for a full response.
  FastAPI's StreamingResponse forwards these tokens to the frontend in real-time.

COMMON FAILURE POINTS:
  - Rate limit (429): Retry with exponential backoff
  - Empty response: Detect and return a helpful fallback message

PRODUCTION CONSIDERATIONS:
  - Use LangSmith for call tracing and debugging
  - Add per-user rate limiting
  - Log all LLM calls for audit
"""

import time
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.config import get_settings
from utils.logger import get_logger
from utils.token_counter import count_tokens, estimate_prompt_cost

logger = get_logger(__name__)
settings = get_settings()


class LLMChain:
    """
    Manages LLM invocation for answer generation using the OpenAI SDK
    pointed at Google's Gemini-compatible endpoint.

    USAGE:
        chain = LLMChain()
        # Non-streaming
        response = await chain.generate(system_prompt, user_prompt)

        # Streaming
        async for token in chain.generate_stream(system_prompt, user_prompt):
            print(token, end="", flush=True)
    """

    def __init__(self):
        # Use langchain_openai.ChatOpenAI but pointed at Google's OpenAI-compatible endpoint.
        # GOOGLE_API_KEY is used as the "openai_api_key" — Google accepts it here.
        self._llm = ChatOpenAI(
            model=settings.LLM_MODEL,               # e.g. "gemini-2.0-flash-lite"
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            openai_api_key=settings.GOOGLE_API_KEY,  # Google key, accepted by compat endpoint
            openai_api_base=settings.GEMINI_API_BASE, # Google's OpenAI-compat URL
            streaming=True,
        )
        logger.info(
            f"LLMChain initialized: model={settings.LLM_MODEL} via {settings.GEMINI_API_BASE}, "
            f"temperature={settings.LLM_TEMPERATURE}"
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """
        Generates a complete (non-streaming) response from Gemini via OpenAI SDK.

        Returns:
            Dict with answer, token counts, estimated cost, and latency.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        input_text = system_prompt + user_prompt
        input_tokens = count_tokens(input_text, settings.LLM_MODEL)
        start_time = time.time()

        try:
            response = await self._invoke_with_retry(messages)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

        answer = response.content
        latency_ms = (time.time() - start_time) * 1000
        output_tokens = count_tokens(answer, settings.LLM_MODEL)
        cost = estimate_prompt_cost(input_tokens, output_tokens, settings.LLM_MODEL)

        logger.info(
            f"LLM response: {input_tokens} in + {output_tokens} out tokens | "
            f"{latency_ms:.0f}ms | ~${cost:.5f}"
        )

        return {
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": cost,
            "latency_ms": latency_ms,
        }

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        """
        Streams the response token-by-token.

        Yields:
            Individual token strings as they arrive from the model.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        logger.debug("Starting streaming response from Gemini...")
        try:
            async for chunk in self._llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"\n[Error: generation failed — {e}]"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _invoke_with_retry(self, messages):
        """Invokes the LLM with automatic retry on transient failures (rate limits, timeouts)."""
        return await self._llm.ainvoke(messages)
