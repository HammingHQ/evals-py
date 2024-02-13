from requests import request
from typing import Any

from .types import HttpClientOptions


class HttpClient:
    api_key: str
    base_url: str

    @staticmethod
    def sanitize_base_url(url: str) -> str:
        return url.strip().rstrip("/")

    def __init__(self, config: HttpClientOptions) -> None:
        self.api_key = config.api_key
        self.base_url = HttpClient.sanitize_base_url(config.base_url)

    def request(self, method: str, url: str, **kwargs) -> Any:
        resp = request(
            method=method,
            url=self.base_url + url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            **kwargs,
        )
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        else:
            return None
