import asyncio
import inspect
from datetime import datetime
from typing import Optional, Dict, Awaitable
from concurrent.futures import ThreadPoolExecutor

from ..types import (
    DatasetItem,
    Experiment,
    ExperimentItem,
    ExperimentItemContext,
    ExperimentStatus,
    InputType,
    MetadataType,
    OutputType,
    Runner,
    RunOptions,
    RunResult,
    ScoreType,
    TracingMode,
    Score,
    CustomScoringConfig,
)
from .api_resource import APIResource
from .scoring import ScoringHelper
from ..utils import get_url_origin

DEFAULT_SCORE_TYPES: list[ScoreType] = [ScoreType.STRING_DIFF]


class ExperimentItems(APIResource):
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

    def end(self, 
        item_context: ExperimentItemContext, 
        output: OutputType,
        scores: Dict[str, Score]
    ):
        item = item_context.item
        start_ts = item_context.startTs
        duration_sec = (datetime.now() - start_ts).total_seconds()
        duration_ms = int(duration_sec * 1000)

        self._client.tracing._flush(item.id)

        self._client.request(
            "PATCH",
            f"/experiments/{item.experimentId}/items/{item.id}",
            json={
                "output": output,
                "scores": {
                    key: value.model_dump() for key, value in scores.items()
                },
                "metrics": {"durationMs": duration_ms},
            },
        )


class Experiments(APIResource):
    _items: ExperimentItems

    @staticmethod
    def generate_name(dataset_name: str) -> str:
        now = datetime.now()
        now.microsecond = 0
        now_str = now.isoformat(sep=" ")
        return f"Experiment for {dataset_name} - {now_str}"

    def __init__(self, client) -> None:
        super().__init__(client)
        self._items = ExperimentItems(client)

    def run(self, opts: RunOptions, run: Runner) -> RunResult:
        self._client.tracing._set_mode(TracingMode.EXPERIMENT)
        
        dataset_id = opts.dataset
        dataset = self._client.datasets.load(dataset_id)

        name = opts.name or Experiments.generate_name(dataset.name)
        scoring = opts.scoring or DEFAULT_SCORE_TYPES
        metadata = opts.metadata or {}

        scoring_helper = ScoringHelper(self._client, scoring)
        scoring_helper.initialize()

        def execute_runner(run: Runner, input: InputType) -> OutputType:
            if inspect.iscoroutinefunction(run):
                return asyncio.run(run(input))
            else:
                return run(input)

        experiment = self._start(
            name, 
            dataset_id, 
            scoring_helper.get_config(), 
            metadata
        )
        url_origin = get_url_origin(self._client.base_url)
        experiment_url = f"{url_origin}/experiments/{experiment.id}"

        def run_item(dataset_item: DatasetItem):
            item_context = self._items.start(experiment, dataset_item)
            output = execute_runner(run, dataset_item.input)
            scores = scoring_helper.score(
                dataset_item.input,
                dataset_item.output,
                output,
            )
            self._items.end(item_context, output, scores)

        try:
            if (opts.parallel):
                if isinstance(opts.parallel, bool):
                    worker_count = None
                elif isinstance(opts.parallel, int):
                    worker_count = opts.parallel
                else:
                    raise ValueError(f"Invalid parallel option: {opts.parallel}")

                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [
                        executor.submit(run_item, dataset_item)
                        for dataset_item in dataset.items
                    ]
                    for future in futures:
                        future.result()
            else:
                for dataset_item in dataset.items:
                    run_item(dataset_item)
            self._end(experiment)

            return RunResult(url=experiment_url)
        except Exception as ex:
            self._end(experiment, status=ExperimentStatus.FAILED)
            raise ex
        
    async def arun(self, opts: RunOptions, run: Runner) -> Awaitable[RunResult]:
        self._client.tracing._set_mode(TracingMode.EXPERIMENT)
        
        dataset_id = opts.dataset
        dataset = self._client.datasets.load(dataset_id)

        name = opts.name or Experiments.generate_name(dataset.name)
        scoring = opts.scoring or DEFAULT_SCORE_TYPES
        metadata = opts.metadata or {}

        scoring_helper = ScoringHelper(self._client, scoring)
        scoring_helper.initialize()

        async def execute_runner(run: Runner, input: InputType) -> OutputType:
            if inspect.iscoroutinefunction(run):
                return await run(input)
            else:
                return run(input)

        experiment = self._start(
            name, 
            dataset_id, 
            scoring_helper.get_config(), 
            metadata
        )
        url_origin = get_url_origin(self._client.base_url)
        experiment_url = f"{url_origin}/experiments/{experiment.id}"

        try:
            for dataset_item in dataset.items:
                item_context = self._items.start(experiment, dataset_item)
                output = await execute_runner(run, dataset_item.input)
                scores = scoring_helper.score(
                    dataset_item.input,
                    dataset_item.output,
                    output,
                )
                self._items.end(item_context, output, scores)
            self._end(experiment)
            return RunResult(url=experiment_url)
        except Exception as ex:
            self._end(experiment, status=ExperimentStatus.FAILED)
            raise ex

    def _start(
        self,
        name: str,
        dataset_id: str,
        scoring: list[ScoreType | CustomScoringConfig],
        metadata: MetadataType,
    ) -> Experiment:
        status = ExperimentStatus.RUNNING
        scoring_obj = [
            s.model_dump() if type(s) == CustomScoringConfig else s for s in scoring
        ]
        resp_data = self._client.request(
            "POST",
            "/experiments",
            json={
                "name": name,
                "dataset": dataset_id,
                "status": status,
                "scoring": scoring_obj,
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
