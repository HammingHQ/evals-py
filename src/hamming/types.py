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
    Literal,
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
    root_seq_id: Optional[int] = None


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
    payload: MonitoringTrace

class TracingMode(Enum):
    OFF = "off"
    MONITORING = "monitoring"
    EXPERIMENT = "experiment"

class Score(BaseModel):
    value: float
    reason: Optional[str]

class FunctionType(str, Enum):
    NUMERIC = "numeric"
    CLASSIFICATION = "classification"

class FunctionAggregateType(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"

class LabelColor(str, Enum):
    GRAY = "gray"
    LIGHT_GREEN = "light-green"
    LIGHT_BLUE = "light-blue"
    AMBER = "amber"
    PURPLE = "purple"
    PINK = "pink"
    GREEN = "green"
    PASTEL_GREEN = "pastel-green"
    YELLOW = "yellow"
    BLUE = "blue"
    RED = "red"

class NumericScoreConfig(BaseModel):
    type: FunctionType = FunctionType.NUMERIC
    aggregate: FunctionAggregateType

class ClassificationScoreConfig(BaseModel):
    type: FunctionType = FunctionType.CLASSIFICATION
    labels: Dict[int, str]
    colors: Optional[Dict[int, LabelColor]]

ScoreConfig = Union[NumericScoreConfig, ClassificationScoreConfig]


class ScorerExecutionType(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"

class ScoreArgs(BaseModel):
    input: InputType
    output: OutputType
    expected: OutputType

class LocalScorer(BaseModel):
    type: ScorerExecutionType = ScorerExecutionType.LOCAL
    score_fn: Callable[[ScoreArgs], Score]

Scorer = LocalScorer

class ScoringFunction(BaseModel):
    name: str
    version: int
    score_config: ScoreConfig
    scorer: Scorer

class CustomScoringConfig(BaseModel):
    id: str
    key_name: str


class RunOptions(BaseModel):
    dataset: str
    name: Optional[str]
    scoring: Optional[list[ScoreType | ScoringFunction]]
    metadata: Optional[MetadataType]
    parallel: Optional[bool | int] = False

class Prompt(BaseModel):
    slug: str

class OpenAIToolChoice(BaseModel):
    modelFamily: Literal["openai"]
    choice: str
    functionName: Optional[str] = None

class AnthropicToolChoice(BaseModel):
    modelFamily: Literal["anthropic"]
    type: str
    tool: Optional[str] = None

class PromptSettings(BaseModel):
    temperature: Optional[float] = None
    maxTokens: Optional[int] = None
    topP: Optional[float] = None
    frequencyPenalty: Optional[float] = None
    presencePenalty: Optional[float] = None
    toolChoice: Optional[OpenAIToolChoice | AnthropicToolChoice] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class PromptContent(BaseModel):
    languageModel: str
    promptSettings: PromptSettings
    chatMessages: list[ChatMessage] = []
    tools: Optional[str] = None

class FullPromptContent(Prompt):
    content: Optional[PromptContent] = None
