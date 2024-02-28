from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import framework


class APIResource:
    _client: framework.Hamming

    def __init__(self, client: framework.Hamming):
        self._client = client
