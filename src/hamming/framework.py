from .types import ClientOptions, HttpClientOptions
from .http_client import HttpClient
from . import resources


DEFAULT_BASE_URL = "https://app.hamming.ai/api/rest"


class Hamming(HttpClient):
    experiments: resources.Experiments
    datasets: resources.Datasets
    tracing: resources.Tracing
    monitoring: resources.Monitoring
    prompts: resources.Prompts
    
    _logger: resources.AsyncLogger

    def __init__(self, config: ClientOptions) -> None:
        super().__init__(
            HttpClientOptions(
                api_key=config.api_key, base_url=config.base_url or DEFAULT_BASE_URL
            )
        )
        self.experiments = resources.Experiments(self)
        self.datasets = resources.Datasets(self)
        self.tracing = resources.Tracing(self)
        self.monitoring = resources.Monitoring(self)
        self.prompts = resources.Prompts(self)

        self._logger = resources.AsyncLogger(self)
        self._logger.start()
        _set_client(self)

def get_client() -> Hamming:
    global _CLIENT
    if _CLIENT is None:
        raise ValueError("Hamming client not initialized")
    return _CLIENT

def _set_client(client: Hamming) -> None:
    global _CLIENT
    _CLIENT = client
