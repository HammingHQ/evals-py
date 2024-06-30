from typing import Optional

from ..types import Prompt, FullPromptContent
from .api_resource import APIResource

class Prompts(APIResource):
    def list(self) -> list[Prompt]:
        resp_data = self._client.request("GET", "/prompts")
        prompts = [Prompt(**prompt_data) for prompt_data in resp_data["prompts"]]
        return prompts

    def get(self, slug: str, label: Optional[str] = None, version: Optional[str] = None) -> Prompt:
        url = f"/prompts/{slug}"
        if label:
            url += f"?label={label}"
        if version:
            url += f"&version={version}"
        resp_data = self._client.request("GET", url)
        prompt = FullPromptContent(**resp_data["prompt"])
        return prompt
