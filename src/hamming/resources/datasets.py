from .api_resource import APIResource

from ..types import DatasetWithItems, Dataset, CreateDatasetOptions


class Datasets(APIResource):
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
