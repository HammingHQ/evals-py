import time
from typing import Optional
from json import dumps

from .framework import Hamming
from .types import GenerationParams, LLMProvider


class OpenAILogger:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    def __init__(self, hamming_client: Hamming):
        self._client = hamming_client

    def log_chat_completion_error(
        self,
        req_kwargs: dict,
        error_message: str,
    ):
        self._log_chat_completion(
            req_kwargs=req_kwargs,
            error=True,
            error_message=error_message,
        )

    def log_chat_completion_success(
        self,
        req_kwargs: dict,
        duration_ms: int,
        resp: ChatCompletion,
    ):
        self._log_chat_completion(
            req_kwargs=req_kwargs,
            duration_ms=duration_ms,
            resp=resp,
        )
    
    def log_chat_completion_success_streaming(
        self,
        req_kwargs: dict,
        duration_ms: int,
        resp_chunks: list[ChatCompletionChunk],
    ):
        self._log_chat_completion(
            req_kwargs=req_kwargs,
            duration_ms=duration_ms,
            resp_chunks=resp_chunks,
        )

    def _log_chat_completion(
        self,
        req_kwargs: dict,
        duration_ms: int = 0,
        resp: Optional[ChatCompletion] = None,
        resp_chunks: Optional[list[ChatCompletionChunk]] = None,
        error: bool = False,
        error_message: Optional[str] = None,
    ):
        model = resp.model if resp else req_kwargs.get("model")
        req_msgs = req_kwargs.get("messages")
        if resp:
            resp_msgs = [c.model_dump() for c in resp.choices]
        elif resp_chunks:
            resp_msgs = [[c.model_dump() for c in chunk.choices] for chunk in resp_chunks]     
        else:
            resp_msgs = []
        self._client.tracing.log_generation(
            GenerationParams(
                input=dumps(req_msgs),
                output=dumps(resp_msgs),
                metadata=GenerationParams.Metadata(
                    provider=LLMProvider.OPENAI,
                    model=model,
                    stream=req_kwargs.get("stream"),
                    max_tokens=req_kwargs.get("max_tokens"),
                    n=req_kwargs.get("n"),
                    seed=req_kwargs.get("seed"),
                    temperature=req_kwargs.get("temperature"),
                    usage=(
                        GenerationParams.Usage(**resp.usage.model_dump())
                        if resp
                        else None
                    ),
                    duration_ms=duration_ms,
                    error=error,
                    error_message=error_message,
                ),
            )
        )


class WrappedSyncCompletions:
    def __init__(self, completions, logger: OpenAILogger):
        self.__original = completions
        self._logger = logger

    def __getattr__(self, name):
        return getattr(self.__original, name)

    def create(self, *args, **kwargs):
        start_ts = time.time()
        is_stream = kwargs.get("stream", False)
        try:
            resp = self.__original.create(*args, **kwargs)
            if is_stream:
                def gen():
                    all_chunks = []
                    try:
                        for chunk in resp:
                            all_chunks.append(chunk)
                            yield chunk
                    except Exception as e:
                        self._logger.log_chat_completion_error(
                            req_kwargs=kwargs,
                            error_message=str(e),
                        )
                        raise e
                    self._logger.log_chat_completion_success_streaming(
                        req_kwargs=kwargs,
                        duration_ms=int((time.time() - start_ts) * 1000),
                        resp_chunks=all_chunks,
                    )
                return gen()
            else:                
                self._logger.log_chat_completion_success(
                    req_kwargs=kwargs,
                    duration_ms=int((time.time() - start_ts) * 1000),
                    resp=resp,
                )
                return resp
        except Exception as e:
            self._logger.log_chat_completion_error(
                req_kwargs=kwargs,
                error_message=str(e),
            )
            raise e


class WrappedAsyncCompletions:
    def __init__(self, completions, logger: OpenAILogger):
        self.__original = completions
        self._logger = logger

    def __getattr__(self, name):
        return getattr(self.__original, name)

    async def create(self, *args, **kwargs):
        start_ts = time.time()
        is_stream = kwargs.get("stream", False)
        try:
            resp = await self.__original.create(*args, **kwargs)
            if is_stream:
                async def gen():
                    all_chunks = []
                    try:
                        async for chunk in resp:
                            all_chunks.append(chunk)
                            yield chunk
                    except Exception as e:
                        self._logger.log_chat_completion_error(
                            req_kwargs=kwargs,
                            error_message=str(e),
                        )
                        raise e
                    self._logger.log_chat_completion_success_streaming(
                        req_kwargs=kwargs,
                        duration_ms=int((time.time() - start_ts) * 1000),
                        resp_chunks=all_chunks,
                    )
                return gen()
            else:                
                self._logger.log_chat_completion_success(
                    req_kwargs=kwargs,
                    duration_ms=int((time.time() - start_ts) * 1000),
                    resp=resp,
                )
                return resp
        except Exception as e:
            self._logger.log_chat_completion_error(
                req_kwargs=kwargs,
                error_message=str(e),
            )
            raise e


def wrap_openai(openai_client, hamming_client):
    from openai import OpenAI, AsyncOpenAI

    logger = OpenAILogger(hamming_client)

    def _instrument_sync_client(openai_client: OpenAI):
        openai_client.chat.completions = WrappedSyncCompletions(
            openai_client.chat.completions, logger
        )

    def _instrument_async_client(openai_client: AsyncOpenAI):
        openai_client.chat.completions = WrappedAsyncCompletions(
            openai_client.chat.completions, logger
        )

    if isinstance(openai_client, OpenAI):
        _instrument_sync_client(openai_client)
    elif isinstance(openai_client, AsyncOpenAI):
        _instrument_async_client(openai_client)
    else:
        print("Unknown client type.")
    return openai_client
