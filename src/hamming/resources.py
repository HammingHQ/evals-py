from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional
import asyncio
import inspect

from .utils import get_url_origin
from .types import (
    RunOptions,
    RunResult,
    Runner,
    DatasetWithItems,
    Dataset,
    CreateDatasetOptions,
    ScoreType,
    MetadataType,
    Experiment,
    ExperimentStatus,
    DatasetItem,
    ExperimentItemContext,
    ExperimentItem,
    InputType,
    OutputType,
    TraceEventType,
    GenerationParams,
    RetrievalParams,
    Document,
)

if TYPE_CHECKING:
    from . import framework


DEFAULT_SCORE_TYPES: list[ScoreType] = [ScoreType.STRING_DIFF]


class APIResource:
    _client: framework.Hamming

    def __init__(self, client: framework.Hamming):
        self._client = client


class ExperimentItems(APIResource):
    def __init__(self, client: framework.Hamming):
        super().__init__(client)

    def start(
        self, experiment: Experiment, dataset_item: DatasetItem
    ) -> ExperimentItemContext:
        resp_data = self._client.request(
            "POST",
            f"/experiments/{experiment.id}/items",
            json={"datasetItemId": dataset_item.id, "output": {}, "metrics": {}},
        )
        item = ExperimentItem(**resp_data["item"])
        item_context = ExperimentItemContext(item=item, startTs=datetime.now())
        return item_context

    def end(self, item_context: ExperimentItemContext, output: OutputType):
        item = item_context.item
        start_ts = item_context.startTs
        duration_sec = (datetime.now() - start_ts).total_seconds()
        duration_ms = int(duration_sec * 1000)

        self._client.tracing._flush(item.id)
        self._client.request(
            "PATCH",
            f"/experiments/{item.experimentId}/items/{item.id}",
            json={"output": output, "metrics": {"durationMs": duration_ms}},
        )


class Experiments(APIResource):
    _items: ExperimentItems

    @staticmethod
    def generate_name(dataset_name: str) -> str:
        now = datetime.now()
        now.microsecond = 0
        now_str = now.isoformat(sep=" ")
        return f"Experiment for {dataset_name} - {now_str}"

    def __init__(self, client: framework.Hamming) -> None:
        super().__init__(client)
        self._items = ExperimentItems(client)

    def run(self, opts: RunOptions, run: Runner) -> RunResult:
        dataset_id = opts.dataset
        dataset = self._client.datasets.load(dataset_id)

        name = opts.name or Experiments.generate_name(dataset.name)
        scoring = opts.scoring or DEFAULT_SCORE_TYPES
        metadata = opts.metadata or {}

        def execute_runner(run: Runner, input: InputType) -> OutputType:
            if inspect.iscoroutinefunction(run):
                return asyncio.run(run(input))
            else:
                return run(input)

        experiment = self._start(name, dataset_id, scoring, metadata)
        url_origin = get_url_origin(self._client.base_url)
        experiment_url = f"{url_origin}/experiments/{experiment.id}"

        try:
            for dataset_item in dataset.items:
                item_context = self._items.start(experiment, dataset_item)
                output = execute_runner(run, dataset_item.input)
                self._items.end(item_context, output)
            self._end(experiment)
            return RunResult(url=experiment_url)
        except Exception as ex:
            self._end(experiment, status=ExperimentStatus.FAILED)
            raise ex

    def _start(
        self,
        name: str,
        dataset_id: str,
        scoring: list[ScoreType],
        metadata: MetadataType,
    ) -> Experiment:
        status = ExperimentStatus.RUNNING
        resp_data = self._client.request(
            "POST",
            "/experiments",
            json={
                "name": name,
                "dataset": dataset_id,
                "status": status,
                "scoring": scoring,
                "metadata": metadata,
            },
        )
        return Experiment(**resp_data["experiment"])

    def _end(
        self,
        experiment: Experiment,
        status: Optional[ExperimentStatus] = ExperimentStatus.FINISHED,
    ):
        self._client.request(
            "PATCH", f"/experiments/{experiment.id}", json={"status": status}
        )


class Datasets(APIResource):
    def __init__(self, client: framework.Hamming) -> None:
        super().__init__(client)

    def load(self, id: str) -> DatasetWithItems:
        resp_data = self._client.request("GET", f"/datasets/{id}")
        return DatasetWithItems(**resp_data["dataset"])

    def list(self) -> list[Dataset]:
        resp_data = self._client.request("GET", "/datasets")
        return [Dataset(**d) for d in resp_data["datasets"]]

    def create(self, create_opts: CreateDatasetOptions) -> Dataset:
        resp_data = self._client.request(
            "POST", "/datasets", json=create_opts.model_dump()
        )
        return Dataset(**resp_data["dataset"])


class Tracing(APIResource):
    _collected: list[TraceEventType] = []
    _current_local_trace_id: int = 0

    def __init__(self, client: framework.Hamming) -> None:
        super().__init__(client)

    def _next_trace_id(self) -> int:
        self._current_local_trace_id += 1
        return self._current_local_trace_id

    def _flush(self, experiment_item_id: str):
        events = self._collected
        self._collected = []

        root_trace: TraceEventType = {
            "id": self._next_trace_id(),
            "experimentItemId": experiment_item_id,
            "event": {"kind": "root"},
        }

        traces: list[TraceEventType] = [root_trace]
        for event in events:
            traces.append(
                {
                    "id": self._next_trace_id(),
                    "experimentItemId": experiment_item_id,
                    "parentId": root_trace["id"],
                    "event": event,
                }
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

    def log(self, trace: TraceEventType):
        self._collected.append(trace)

    def log_generation(self, params: GenerationParams):
        self.log(Tracing._generation_event(params))

    def log_retrieval(self, params: RetrievalParams):
        self.log(Tracing._retrieval_event(params))
