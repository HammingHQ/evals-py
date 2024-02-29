import logging

from . import APIResource
from ..types import (
    ExperimentTrace,
    MonitoringTrace,
    TraceEventType,
    GenerationParams,
    RetrievalParams,
    Document,
    LogMessage,
    LogMessageType,
    TracingMode,
)

log = logging.getLogger(__name__)


class Tracing(APIResource):
    _collected_events: list[TraceEventType] = []
    _current_local_trace_id: int = 0

    _mode: TracingMode = TracingMode.OFF

    def _set_mode(self, mode: TracingMode):
        self._mode = mode

    def _next_trace_id(self) -> int:
        self._current_local_trace_id += 1
        return self._current_local_trace_id

    def _flush(self, experiment_item_id: str):
        if self._mode != TracingMode.EXPERIMENT:
            log.warning("Tracing mode must be set to <experiment>!")
            return
        events = self._collected_events
        self._collected_events = []

        root_trace = ExperimentTrace(
            id=self._next_trace_id(),
            experimentItemId=experiment_item_id,
            event={"kind": "root"},
        )

        traces: list[ExperimentTrace] = [root_trace]
        for event in events:
            traces.append(
                ExperimentTrace(
                    id=self._next_trace_id(),
                    experimentItemId=experiment_item_id,
                    parentId=root_trace.id,
                    event=event,
                )
            )

        self._client.request("POST", "/traces", json={"traces": traces})

    @staticmethod
    def _generation_event(params: GenerationParams) -> TraceEventType:
        event = params.model_dump()
        event["kind"] = "llm"
        return event

    @staticmethod
    def _retrieval_event(params: RetrievalParams) -> TraceEventType:
        def normalize_document(doc: Document | str) -> Document:
            if isinstance(doc, str):
                return Document(pageContent=doc, metadata={})
            return doc

        params.results = [normalize_document(r) for r in params.results]
        event = params.model_dump()
        event["kind"] = "vector"
        return event

    def _log_live_trace(self, trace: MonitoringTrace):
        if self._mode != TracingMode.MONITORING:
            log.warning("Tracing mode must be set to <monitoring>!")
            return
        self._client._logger.log(
            LogMessage(message_type=LogMessageType.Monitoring, message_payload=trace)
        )

    def log(self, trace: TraceEventType):
        if self._mode == TracingMode.MONITORING:
            context = self._client.monitoring._get_tracing_context()
            self._log_live_trace(
                MonitoringTrace(
                    session_id=context.session_id,
                    seq_id=context.seq_id,
                    parent_seq_id=context.parent_seq_id,
                    event=trace,
                )
            )
        elif self._mode == TracingMode.EXPERIMENT:
            self._collected_events.append(trace)
        else:
            log.warning("Attempt to send a log trace, but tracing mode is off!")

    def log_generation(self, params: GenerationParams):
        self.log(Tracing._generation_event(params))

    def log_retrieval(self, params: RetrievalParams):
        self.log(Tracing._retrieval_event(params))
