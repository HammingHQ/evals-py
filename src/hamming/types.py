from datetime import datetime
from enum import Enum
from pydantic import BaseModel
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Mapping,
    Optional,
    TypeAlias,
    Union,
)


class ClientOptions(BaseModel):
    api_key: str
    base_url: Optional[str] = None


class HttpClientOptions(BaseModel):
    api_key: str
    base_url: str


class ScoreType(str, Enum):
    ACCURACY_AI = "accuracy_ai"
    FACTS_COMPARE = "facts_compare"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_PRECISION = "context_precision"
    HALLUCINATION = "hallucination"
    STRING_DIFF = "string_diff"
    REFUSAL = "refusal"
    SQL_AST = "sql_ast"


InputType: TypeAlias = Dict
OutputType: TypeAlias = Dict
MetadataType: TypeAlias = Dict


class RunOptions(BaseModel):
    dataset: str
    name: Optional[str]
    scoring: Optional[list[ScoreType]]
    metadata: Optional[MetadataType]


Runner: TypeAlias = Union[
    Callable[[InputType], OutputType], Callable[[InputType], Awaitable[OutputType]]
]


class RunResult(BaseModel):
    url: str


class DatasetItemValue(BaseModel):
    input: InputType
    output: OutputType
    metadata: MetadataType


class DatasetItem(DatasetItemValue):
    id: str


class Dataset(BaseModel):
    id: str
    name: str
    description: Optional[str]


class DatasetWithItems(Dataset):
    items: list[DatasetItem]


class CreateDatasetOptions(BaseModel):
    name: str
    description: Optional[str]
    items: list[DatasetItemValue]


class ExperimentStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SCORING = "SCORING"
    SCORING_FAILED = "SCORING_FAILED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class ExperimentItemStatus(str, Enum):
    CREATED = "CREATED"
    SCORING = "SCORING"
    SCORED = "SCORED"
    SCORING_FAILED = "SCORING_FAILED"
    FAILED = "FAILED"


class Experiment(BaseModel):
    id: str
    name: str
    description: Optional[str]
    datasetId: str
    datasetVersionId: Optional[int]
    status: ExperimentStatus


class ExperimentItemMetrics(BaseModel):
    durationMs: Optional[int] = 0


class ExperimentItem(BaseModel):
    id: str
    experimentId: str
    datasetItemId: str
    output: OutputType
    metrics: ExperimentItemMetrics
    status: ExperimentItemStatus


class ExperimentItemContext(BaseModel):
    item: ExperimentItem
    startTs: datetime


TraceEventType: TypeAlias = Mapping[str, Any]


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


class GenerationParams(BaseModel):
    class Usage(BaseModel):
        completion_tokens: int
        prompt_tokens: int
        total_tokens: int

    class Metadata(BaseModel):
        provider: Optional[Union[LLMProvider, str]] = None
        model: Optional[str] = None
        stream: Optional[bool] = None
        max_tokens: Optional[int] = None
        n: Optional[int] = None
        seed: Optional[int] = None
        temperature: Optional[float] = None
        usage: Optional["GenerationParams.Usage"] = None
        duration_ms: Optional[int] = None
        error: bool = False
        error_message: Optional[str] = None

    input: Optional[str] = None
    output: Optional[str] = None
    metadata: Optional[Metadata] = None


class Document(BaseModel):
    pageContent: str
    metadata: Mapping[str, Any]


class RetrievalParams(BaseModel):
    class Metadata(BaseModel):
        engine: Optional[str] = None

    query: Optional[str] = None
    results: Union[list[Document], list[str]]
    metadata: Optional[Metadata] = None


class ExperimentTrace(BaseModel):
    id: int
    experimentItemId: str
    parentId: Optional[int] = None
    event: TraceEventType


class MonitoringTraceContext(BaseModel):
    session_id: str
    seq_id: int
    parent_seq_id: Optional[int] = None


class MonitoringTrace(MonitoringTraceContext):
    event: TraceEventType


class MonitoringItemStatus(str, Enum):
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class LogMessageType(int, Enum):
    Monitoring = 1


class LogMessage(BaseModel):
    type: LogMessageType
    payload: Union[MonitoringTrace]

class TracingMode(Enum):
    OFF = "off"
    MONITORING = "monitoring"
    EXPERIMENT = "experiment"
