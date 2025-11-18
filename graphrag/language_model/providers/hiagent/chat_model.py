import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

from graphrag.language_model.providers.hiagent.client import HiAgentLLMClient
from graphrag.language_model.response.base import (BaseModelOutput,
                                                   BaseModelResponse,
                                                   ModelResponse)


class HiAgentChatLLM:
    """HiAgent LLM provider for GraphRAG."""

    def __init__(
        self,
        conversation_inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the HiAgent LLM adapter.

        Args:
            api_url: HiAgent API URL
            api_key: HiAgent
        """
        self.client = HiAgentLLMClient(
            api_url="https://agent.hust.edu.cn",
            api_key=os.getenv("HIAGENT_APIKEY"),
            user_agent="GraphRAG/1.0",
        )
        self.user_id = "graphrag_user"
        self.conversation_inputs = conversation_inputs or {}
        self._conversation_created = False

    def _ensure_conversation(self):
        """Ensure that the conversation has been created."""
        if not self._conversation_created:
            self.client.create_conversation(
                user_id=self.user_id,
                inputs=self.conversation_inputs,
            )
            self._conversation_created = True

    async def achat(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Send an asynchronous chat request.

        Args:
            prompt: User input prompt
            history: Conversation history (managed automatically by HiAgent)
            **kwargs: Additional parameters

        Returns:
            ModelResponse: Model response
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.chat(prompt, history, **kwargs)
        )

    async def achat_stream(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream an asynchronous chat response.

        Args:
            prompt: User input prompt
            history: Conversation history
            **kwargs: Additional parameters

        Yields:
            str: Streamed text chunks from the response
        """
        self._ensure_conversation()
        query_extends = kwargs.get("query_extends")

        loop = asyncio.get_event_loop()

        def stream_generator():
            for data in self.client.chat_query_streaming(
                query=prompt,
                query_extends=query_extends,
                chunk_size=kwargs.get("chunk_size", 1),
            ):
                if data.get("event") == "message":
                    answer = data.get("answer", "")
                    if answer:
                        yield answer

        for chunk in await loop.run_in_executor(None, list, stream_generator()):
            yield chunk

    def chat(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs,
    ) -> ModelResponse:
        """
        Send a synchronous chat request.

        Args:
            prompt: User input prompt
            history: Conversation history (managed automatically by HiAgent)
            **kwargs: Additional parameters

        Returns:
            ModelResponse: Model response
        """
        self._ensure_conversation()

        query_extends = kwargs.get("query_extends")
        response = self.client.chat_query_blocking(
            query=prompt,
            query_extends=query_extends,
        )

        content = response.get("answer", "")

        parsed_json = None
        if kwargs.get("json", False):
            try:
                import json

                parsed_json = json.loads(content)
            except json.JSONDecodeError:
                pass

        return BaseModelResponse(
            output=BaseModelOutput(content=content),
            parsed_response=parsed_json,
        )

    def chat_stream(
        self,
        prompt: str,
        history: list | None = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Stream a synchronous chat response.

        Args:
            prompt: User input prompt
            history: Conversation history
            **kwargs: Additional parameters

        Yields:
            str: Streamed text chunks from the response
        """
        self._ensure_conversation()
        query_extends = kwargs.get("query_extends")

        for data in self.client.chat_query_streaming(
            query=prompt,
            query_extends=query_extends,
            chunk_size=kwargs.get("chunk_size", 1),
        ):
            if data.get("event") == "message":
                answer = data.get("answer", "")
                if answer:
                    yield answer
