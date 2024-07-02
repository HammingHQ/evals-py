from dataclasses import dataclass
import contextvars
from typing import Optional
import uuid
import threading
import time
import logging

from .api_resource import APIResource
from ..types import (
    MonitoringTrace, 
    MonitoringTraceContext, 
    MonitoringItemStatus, 
    TracingMode
)

logger = logging.getLogger(__name__)


@dataclass
class Session:
    id: str = str(uuid.uuid4())
    seq_id: int = 0


class Monitoring(APIResource):
    _lock = threading.RLock()
    _session: Optional[Session] = None
    _current_tree = contextvars.ContextVar("hamming_monitoring_tree", default=None)

    def start(self):
        with self._lock:
            if not self._session:
                self._session = Session()
            self._client.tracing._set_mode(TracingMode.MONITORING)

    def stop(self):
        with self._lock:
            self._session = None
            self._client.tracing._set_mode(TracingMode.OFF)

    def start_item(
        self, 
        input: Optional[dict] = {}, 
        name: Optional[str] = None, 
        metadata: Optional[dict] = {}
    ):
        (session_id, seq_id) = self._next_seq_id()
        current_tree = self._current_tree.get()
        if not current_tree:
            current_tree = MonitoringTree()
        parent_item = current_tree.get_current_item()
        parent_seq_id = parent_item._seq_id if parent_item else None
        root_item = current_tree.get_root_item()
        root_seq_id = root_item._seq_id if root_item else None
        item = MonitoringItem(
            self, 
            session_id, 
            seq_id, 
            parent_seq_id, 
            root_seq_id
        )
        item._start(input, name, metadata)
        current_tree.add_child(item)
        self._current_tree.set(current_tree)
        return item

    def _end_item(self, trace: MonitoringTrace):
        current_tree: MonitoringTree = self._current_tree.get()
        if not current_tree:
            logger.warn("Missing monitoring tree")
        else:
            try:
                current_tree.pop_child()
            except IndexError:
                logger.warn("Missing monitoring item")
        self._client.tracing._log_live_trace(trace)

    def _get_tracing_context(self) -> MonitoringTraceContext:
        with self._lock:
            if not self._session:
                raise Exception("Monitoring not started")
            current_tree: MonitoringTree = self._current_tree.get()
            current_item = None
            root_item = None
            if current_tree:
                current_item = current_tree.get_current_item()
                root_item = current_tree.get_root_item()
            (session_id, seq_id) = self._next_seq_id()
            return MonitoringTraceContext(
                session_id=session_id,
                seq_id=seq_id,
                parent_seq_id=current_item._seq_id if current_item else None,
                root_seq_id=root_item._seq_id if root_item else None,
            )

    def _next_seq_id(self) -> tuple[str, int]:
        with self._lock:
            if not self._session:
                raise Exception("Monitoring not started")
            self._session.seq_id += 1
            return (self._session.id, self._session.seq_id)


class MonitoringItem:
    _session_id: str
    _seq_id: int
    _parent_seq_id: Optional[int]
    _root_seq_id: Optional[int]
    _input: Optional[dict] = None
    _output: Optional[dict] = None
    _name: Optional[str] = None
    _metadata: Optional[dict] = None
    _metrics: dict
    _status: MonitoringItemStatus
    _error_message: Optional[str] = None
    _start_ts: Optional[float]

    def __init__(
        self, 
        monitoring: Monitoring, 
        session_id: str, 
        seq_id: int, 
        parent_seq_id: Optional[int] = None,
        root_seq_id: Optional[int] = None
    ):
        self._monitoring = monitoring
        self._session_id = session_id
        self._seq_id = seq_id
        self._parent_seq_id = parent_seq_id
        self._root_seq_id = root_seq_id
        self._metrics = {}

    def set_input(self, input: dict):
        self._input = input

    def set_output(self, output: dict):
        self._output = output

    def set_metadata(self, metadata: dict):
        self._metadata = metadata

    def _start(
        self, 
        input: Optional[dict] = {}, 
        name: Optional[str] = None, 
        metadata: Optional[dict] = {}
    ):
        self._input = input
        self._name = name
        self._metadata = metadata
        self._start_ts = time.time()
        self._status = MonitoringItemStatus.STARTED

    def _end(self, error: bool = False, error_message: Optional[str] = None):
        if self._has_ended():
            return
        self._metrics["duration_ms"] = int((time.time() - self._start_ts) * 1000)
        self._status = (
            MonitoringItemStatus.FAILED if error else MonitoringItemStatus.COMPLETED
        )
        self._error_message = error_message
        self._monitoring._end_item(self._to_trace())

    def _has_ended(self) -> bool:
        return self._status in [
            MonitoringItemStatus.COMPLETED,
            MonitoringItemStatus.FAILED,
        ]

    def _to_trace(self) -> MonitoringTrace:
        event_kind = "root" if self._parent_seq_id is None else "span"
        return MonitoringTrace(
            session_id=self._session_id,
            seq_id=self._seq_id,
            parent_seq_id=self._parent_seq_id,
            root_seq_id=self._root_seq_id,
            event={
                "kind": event_kind,
                "input": self._input,
                "output": self._output,
                "name": self._name,
                "metadata": self._metadata,
                "metrics": self._metrics,
                "status": self._status,
                "error_message": self._error_message,
            },
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self._end(error=True, error_message=exc_value)
        else:
            self._end()


class MonitoringTree:
    path: list[MonitoringItem]

    def __init__(self):
        self.path = []

    def add_child(self, item: MonitoringItem):
        self.path.append(item)

    def pop_child(self):
        return self.path.pop()
    
    def get_current_item(self) -> Optional[MonitoringItem]:
        return self.path[-1] if self.path and len(self.path) > 0 else None

    def get_root_item(self) -> Optional[MonitoringItem]:
        return self.path[0] if self.path and len(self.path) > 0 else None
