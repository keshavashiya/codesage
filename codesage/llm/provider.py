"""LangChain LLM provider for code intelligence."""

from typing import Any, Callable, Dict, List, Optional, TypeVar

from langchain_ollama import ChatOllama

F = TypeVar("F", bound=Callable[..., Any])
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from codesage.utils.config import Config, LLMConfig
from codesage.utils.rate_limiter import RateLimiter
from codesage.utils.retry import retry_with_backoff
from codesage.utils.logging import get_logger

logger = get_logger("llm.provider")


def _build_model_kwargs(config: LLMConfig) -> Dict[str, Any]:
    """Build comprehensive model parameters for Ollama.

    Args:
        config: LLM configuration

    Returns:
        Dictionary of model parameters for Ollama
    """
    params: Dict[str, Any] = {
        # Core parameters
        "num_ctx": config.context_window,
        "num_predict": config.num_predict,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "repeat_penalty": config.repeat_penalty,
        "repeat_last_n": config.repeat_last_n,
    }

    # Add model-specific parameters
    params.update(config.model_parameters)

    return params


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    pass


class LLMConnectionError(LLMProviderError):
    """Raised when unable to connect to LLM service."""

    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when LLM request times out."""

    pass


class LLMProvider:
    """LangChain-based LLM provider.

    Wraps LangChain's LLM interfaces for code analysis and generation.
    Supports Ollama by default, with optional OpenAI/Anthropic.

    Features:
        - Configurable timeouts to prevent hangs
        - Automatic retry with exponential backoff
        - Rate limiting to prevent overwhelming the service
    """

    def __init__(self, config: LLMConfig, requests_per_minute: int = 60):
        """Initialize the LLM provider.

        Args:
            config: LLM configuration
            requests_per_minute: Rate limit for LLM calls
        """
        self.config = config
        self._llm: Optional[BaseChatModel] = None
        # Use config-based rate limiting with enhanced token bucket
        self._rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            bucket_size=config.token_bucket_size,
            refill_rate=config.token_bucket_refill_rate,
        )
        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize the LangChain LLM based on provider."""
        timeout = self.config.request_timeout

        if self.config.provider == "ollama":
            # Build model parameters for enhanced Ollama configuration
            model_kwargs = _build_model_kwargs(self.config)

            self._llm = ChatOllama(
                model=self.config.model,
                base_url=self.config.base_url,
                temperature=self.config.temperature,
                timeout=timeout,  # Add timeout
                # Pass additional parameters via model_kwargs
                model_kwargs=model_kwargs,
                stop=self.config.stop_sequences,
            )
        elif self.config.provider == "openai":
            try:
                from langchain_openai import ChatOpenAI

                self._llm = ChatOpenAI(
                    model=self.config.model,
                    api_key=self.config.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=timeout,
                    max_retries=0,  # We handle retries ourselves
                )
            except ImportError:
                raise ImportError(
                    "OpenAI support requires langchain-openai. "
                    "Install with: pipx inject pycodesage 'pycodesage[openai]' (or pip install 'pycodesage[openai]')"
                )
        elif self.config.provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic

                self._llm = ChatAnthropic(
                    model=self.config.model,
                    api_key=self.config.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=timeout,
                    max_retries=0,  # We handle retries ourselves
                )
            except ImportError:
                raise ImportError(
                    "Anthropic support requires langchain-anthropic. "
                    "Install with: pipx inject pycodesage 'pycodesage[anthropic]' (or pip install 'pycodesage[anthropic]')"
                )
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    @property
    def llm(self) -> BaseChatModel:
        """Get the underlying LangChain LLM."""
        if self._llm is None:
            raise RuntimeError("LLM not initialized")
        return self._llm

    def _create_retry_decorator(self) -> Callable[[F], F]:
        """Create a retry decorator based on config settings."""
        return retry_with_backoff(
            max_retries=self.config.max_retries,
            base_delay=1.0,
            max_delay=30.0,
            exceptions=(
                ConnectionError,
                TimeoutError,
                OSError,
                LLMConnectionError,
                LLMTimeoutError,
            ),
            on_retry=lambda e, attempt: logger.warning(
                f"LLM call failed, retrying (attempt {attempt + 1}): {e}"
            ),
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature

        Returns:
            Generated text response

        Raises:
            LLMProviderError: If generation fails after retries
        """
        # Apply rate limiting
        self._rate_limiter.acquire()

        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        # Use retry decorator for the actual LLM call
        @self._create_retry_decorator()
        def _invoke_llm():
            try:
                response = self._llm.invoke(messages)
                return (
                    response.content if hasattr(response, "content") else str(response)
                )
            except TimeoutError as e:
                raise LLMTimeoutError(f"LLM request timed out: {e}") from e
            except ConnectionError as e:
                raise LLMConnectionError(f"Failed to connect to LLM: {e}") from e
            except Exception as e:
                # Log unexpected errors but let them propagate for retry
                logger.error(f"Unexpected LLM error: {type(e).__name__}: {e}")
                raise

        try:
            return _invoke_llm()
        except Exception as e:
            logger.error(f"LLM generation failed after retries: {e}")
            raise LLMProviderError(f"Failed to generate response: {e}") from e

    def chat(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Chat completion with message history.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            AI response text
        """
        # Apply rate limiting
        self._rate_limiter.acquire()

        lc_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        @self._create_retry_decorator()
        def _invoke_chat():
            try:
                response = self._llm.invoke(lc_messages)
                return (
                    response.content if hasattr(response, "content") else str(response)
                )
            except TimeoutError as e:
                raise LLMTimeoutError(f"Chat request timed out: {e}") from e
            except ConnectionError as e:
                raise LLMConnectionError(f"Failed to connect to LLM: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected chat error: {type(e).__name__}: {e}")
                raise

        try:
            return _invoke_chat()
        except Exception as e:
            logger.error(f"Chat completion failed after retries: {e}")
            raise LLMProviderError(f"Failed to complete chat: {e}") from e

    def stream_chat(self, messages: List[Dict[str, str]]):
        """Stream chat completion tokens.

        Yields:
            String chunks as they arrive from the LLM.
        """
        self._rate_limiter.acquire()

        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        try:
            for chunk in self._llm.stream(lc_messages):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    yield content
        except TimeoutError as e:
            raise LLMTimeoutError(f"Stream request timed out: {e}") from e
        except ConnectionError as e:
            raise LLMConnectionError(
                f"Failed to connect to LLM for streaming: {e}"
            ) from e

    def is_available(self) -> bool:
        """Check if the LLM is available and configured."""
        try:
            # Try a simple test with short timeout
            self._llm.invoke([HumanMessage(content="test")])
            return True
        except Exception as e:
            logger.debug(f"LLM availability check failed: {e}")
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get LLM provider metrics.

        Returns:
            Dictionary with rate limiter and usage metrics
        """
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "rate_limiter": self._rate_limiter.get_metrics(),
        }


def create_llm_provider(config: Config) -> LLMProvider:
    """Factory function to create an LLM provider from config.

    Args:
        config: Full CodeSage config

    Returns:
        Configured LLM provider
    """
    return LLMProvider(config.llm)
