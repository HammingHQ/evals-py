from .types import ClientOptions, HttpClientOptions
from .http_client import HttpClient
from . import resources


DEFAULT_BASE_URL = "https://app.hamming.ai/api/rest"


class Hamming(HttpClient):
    experiments: resources.Experiments
    datasets: resources.Datasets
    tracing: resources.Tracing
    monitoring: resources.Monitoring

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

        self._logger = resources.AsyncLogger(self)
        self._logger.start()
