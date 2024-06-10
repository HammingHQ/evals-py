from __future__ import annotations

import atexit
import logging
import queue
import threading
from typing import TYPE_CHECKING

from ..types import LogMessage
from .api_resource import APIResource

if TYPE_CHECKING:
    from .. import framework


LOG_BATCH_SIZE = 512


log = logging.getLogger(__name__)


class AsyncLogger(APIResource):
    _queue: queue.Queue[LogMessage]
    _thread: threading.Thread
    _stop_event: threading.Event

    def __init__(self, _client: framework.Hamming):
        super().__init__(_client)
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run_queue, daemon=True)
        self._stop_event = threading.Event()
        atexit.register(self.stop)

    def log(self, message: LogMessage):
        self._queue.put(message)

    def start(self):
        log.info("Starting logger thread..")
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        log.info("Waiting for logger thread to exit..")
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join()
        log.info("Logger thread exited!")

    def _drain_queue(self):
        drained_msgs = []
        while self._queue.qsize() > 0 and len(drained_msgs) < LOG_BATCH_SIZE:
            try:
                msg = self._queue.get_nowait()
                if msg is not None:
                    drained_msgs.append(msg)
            except queue.Empty:
                break
        return drained_msgs

    def _process_queue(self) -> bool:
        log.debug("Processing loop started..")

        msg = self._queue.get()
        if msg is None:
            log.info("Received stop signal from queue!")
            self._queue.task_done()
            return True
        msgs_to_process = [msg]
        msgs_to_process.extend(self._drain_queue())

        log.debug(f"Processing {len(msgs_to_process)} message(s) from the queue")

        self._publish(msgs_to_process)
        for msg in msgs_to_process:
            self._queue.task_done()
        log.debug("Processing loop done!")
        return False

    def _run_queue(self):
        while not self._stop_event.is_set():
            done = self._process_queue()
            if done:
                # Skip last loop if we're shutting down
                return
        # Last processing loop to drain the queue
        self._process_queue()

    def _publish(self, msgs: list[LogMessage]):
        log.debug(f"Publishing {len(msgs)} messages..")
        try:
            logs = [m.model_dump(exclude_none=True) for m in msgs]
            self._client.request("POST", "/logs", json={"logs": logs})
            log.debug(f"Published {len(msgs)} messages!")
        except Exception as e:
            log.error(f"Failed to publish messages: {e}")
