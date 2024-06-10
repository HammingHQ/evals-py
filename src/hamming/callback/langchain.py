from collections import defaultdict
from typing import Dict, Any, Optional, List, Union, Sequence
from uuid import UUID
from contextvars import copy_context, Context
import logging
import json

from .. import framework
from ..resources import monitoring
from ..types import GenerationParams, RetrievalParams

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_core.messages import (
        BaseMessage,
        HumanMessage,
        AIMessage,
        SystemMessage,
        ToolMessage,
        FunctionMessage,
        ChatMessage
    )
    from langchain.schema.document import Document
except ImportError:
    raise ModuleNotFoundError(
        "Please install langchain in order to use the Langchain integration."
    )


log = logging.getLogger(__name__)


class LangchainCallbackHandler(BaseCallbackHandler):
    hamming: framework.Hamming
    run_items: Dict[UUID, monitoring.MonitoringItem] = defaultdict(None)
    run_ctx: Dict[UUID, Context] = defaultdict(None)
    run_parent: Dict[UUID, UUID] = defaultdict(None)
    run_llm_input: Dict[UUID, str] = defaultdict(None)
    run_llm_serialized: Dict[UUID, Dict[str, Any]] = defaultdict(None)
    run_retriever_query: Dict[UUID, str] = defaultdict(None)
    run_retriever_serialized: Dict[UUID, Dict[str, Any]] = defaultdict(None)

    def __init__(self, hamming: framework.Hamming):
        self.hamming = hamming
        self.hamming.monitoring.start()
        log.debug("LangchainCallbackHandler initialized!")

    def on_chain_start(
        self, 
        serialized: Dict[str, Any], 
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        if parent_run_id is not None:
            self.run_parent[run_id] = parent_run_id
            return
        self.run_items[run_id] = self.hamming.monitoring.start_item(input=inputs)
        self.run_ctx[run_id] = copy_context()

    def on_chain_end(
        self,
        outputs: Union[str, Dict[str, Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ):
        if parent_run_id is not None:
            if run_id in self.run_parent:
                del self.run_parent[run_id]
            return
        
        monitoring_item = self.run_items.pop(run_id)
        if monitoring_item is None:
            log.warning(f"No monitoring item found for run_id: {run_id}")
            return

        if isinstance(outputs, str):
            monitoring_item.set_output({
                "response": outputs
            })
        else:
            monitoring_item.set_output(outputs)
        monitoring_item._end()

    def on_chain_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ):
        monitoring_item = self.run_items.pop(run_id)
        if monitoring_item is None:
            log.warning(f"No monitoring item found for run_id: {run_id}")
            return
        monitoring_item.set_output({})
        monitoring_item._end(True, str(error))

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        flatten_messages = [message for sublist in messages for message in sublist]
        message_dicts = [self._convert_message(message) for message in flatten_messages]
        self.run_llm_input[run_id] = json.dumps(message_dicts)
        self.run_llm_serialized[run_id] = serialized

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        self.run_llm_input[run_id] = json.dumps(prompts)
        self.run_llm_serialized[run_id] = serialized

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        llm_input = self.run_llm_input.pop(run_id)
        llm_serialized = self.run_llm_serialized.pop(run_id)

        metadata=GenerationParams.Metadata()
        if "openai" in llm_serialized.get("id", []):
            metadata.provider = "openai"
            kw = llm_serialized.get("kwargs", {})
            metadata.model = kw.get("model_name", None)
            metadata.temperature = kw.get("temperature", None)
            metadata.max_tokens = kw.get("max_tokens", None)
            metadata.n = kw.get("n", None)

        llm_params = GenerationParams(
            input=llm_input,
            output=response.generations[-1][-1].text,
            metadata=metadata,
        )

        top_parent_run_id = self._find_top_parent_run_id(parent_run_id)
        ctx = self.run_ctx.get(top_parent_run_id, None)
        if ctx is not None:
            ctx.run(self.hamming.tracing.log_generation, llm_params)
        else:
            self.hamming.tracing.log_generation(params=llm_params)

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        self.run_retriever_query[run_id] = query
        self.run_retriever_serialized[run_id] = serialized

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        retriever_query = self.run_retriever_query.pop(run_id)
        retriever_serialized = self.run_retriever_serialized.pop(run_id)
        metadata = RetrievalParams.Metadata(
            engine="langchain" # TODO: detect engine
        )
        retrieval_params = RetrievalParams(
            query=retriever_query,
            results=[document.page_content for document in documents],
            metadata=metadata,
        )
        top_parent_run_id = self._find_top_parent_run_id(parent_run_id)
        ctx = self.run_ctx.get(top_parent_run_id, None)
        if ctx is not None:
            ctx.run(self.hamming.tracing.log_retrieval, retrieval_params)
        else:
            self.hamming.tracing.log_retrieval(params=retrieval_params)

    def _find_top_parent_run_id(self, run_id: UUID):
        parent_run_id = self.run_parent.get(run_id, None)
        if parent_run_id is not None:
            return self._find_top_parent_run_id(parent_run_id)
        return run_id
    

    def _convert_message(self, message: BaseMessage) -> Dict[str, Any]:
        if isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, ToolMessage):
            message_dict = {"role": "tool", "content": message.content}
        elif isinstance(message, FunctionMessage):
            message_dict = {"role": "function", "content": message.content}
        elif isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_dict
