import time
from typing import Optional
from json import dumps

from .framework import Hamming
from .types import GenerationParams, LLMProvider


class OpenAILogger:
    from openai.types.chat import ChatCompletion

    def __init__(self, hamming_client: Hamming):
        self._client = hamming_client

    def log_chat_completion(
        self,
        duration_ms: int,
        req_kwargs: dict,
        resp: Optional[ChatCompletion] = None,
        error: bool = False,
        error_message: Optional[str] = None,
    ):
        model = resp.model if resp else req_kwargs.get("model")
        req_msgs = req_kwargs.get("messages")
        resp_msgs = [c.model_dump() for c in resp.choices] if resp else []
        self._client.tracing.log_generation(
            GenerationParams(
                input=dumps(req_msgs),
                output=dumps(resp_msgs) if resp else None,
                metadata=GenerationParams.Metadata(
                    provider=LLMProvider.OPENAI,
                    model=model,
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
        error = False
        error_message = None
        resp = None
        try:
            resp = self.__original.create(*args, **kwargs)
        except Exception as e:
            error = True
            error_message = str(e)
            raise e
        finally:
            end_ts = time.time()
            duration_ms = int((end_ts - start_ts) * 1000)
            self._logger.log_chat_completion(
                duration_ms=duration_ms,
                req_kwargs=kwargs,
                resp=resp,
                error=error,
                error_message=error_message,
            )
        return resp


class WrappedAsyncCompletions:
    def __init__(self, completions, logger: OpenAILogger):
        self.__original = completions
        self._logger = logger

    def __getattr__(self, name):
        return getattr(self.__original, name)

    async def create(self, *args, **kwargs):
        start_ts = time.time()
        error = False
        error_message = None
        resp = None
        try:
            resp = await self.__original.create(*args, **kwargs)
        except Exception as e:
            error = True
            error_message = str(e)
            raise e
        finally:
            end_ts = time.time()
            duration_ms = int((end_ts - start_ts) * 1000)
            self._logger.log_chat_completion(
                duration_ms=duration_ms,
                req_kwargs=kwargs,
                resp=resp,
                error=error,
                error_message=error_message,
            )
        return resp


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
