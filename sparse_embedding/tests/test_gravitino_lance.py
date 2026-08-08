from __future__ import annotations

import pytest

from sparse_embedding.checkpoint.catalog import TableHandle
from sparse_embedding.checkpoint.gravitino_lance import (
    GravitinoCatalogAdmin,
    GravitinoLanceStorage,
    LanceRestCatalog,
    LanceRestError,
    SdkConfig,
)


class _Response:
    ok = True
    content = b"{}"
    status_code = 200
    text = ""

    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError("unexpected HTTP response")


class _RestSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, url: str, **kwargs: object) -> _Response:
        self.calls.append((method, url))
        return _Response({})

    def close(self) -> None:
        pass


class _CatalogSession:
    def __init__(self, get_status_code: int = 200) -> None:
        self._get_status_code = get_status_code
        self.urls: list[str] = []
        self.created_name: str | None = None

    def get(self, url: str, **kwargs: object) -> _Response:
        self.urls.append(url)
        return _Response(
            {"catalog": {"properties": {}}},
            status_code=self._get_status_code,
        )

    def post(self, url: str, **kwargs: object) -> _Response:
        payload = kwargs["json"]
        assert isinstance(payload, dict)
        name = payload["name"]
        assert isinstance(name, str)
        self.created_name = name
        return _Response({"catalog": {}})

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("api_base_uri", "request_url"),
    (
        (
            "https://vendor.example/gateway/lance",
            "https://vendor.example/gateway/lance/namespace/list",
        ),
        (
            "https://vendor.example/gateway/lance/v1/",
            "https://vendor.example/gateway/lance/v1/namespace/list",
        ),
    ),
)
def test_lance_rest_uses_configured_complete_api_root(
    api_base_uri: str, request_url: str
) -> None:
    session = _RestSession()
    client = LanceRestCatalog(
        SdkConfig.for_rest(lance_rest_api_base_uri=api_base_uri),
        session=session,
    )

    client.list_namespaces()

    assert session.calls == [("GET", request_url)]


def test_catalog_admin_uses_independent_configured_control_api_root() -> None:
    session = _CatalogSession()
    config = SdkConfig.for_catalog_admin(
        metalake="training",
        catalog_name="embeddings",
        catalog_location="s3://bucket/embeddings",
        s3_endpoint="https://object-store.example",
        s3_region="region-1",
        s3_access_key="test-access-key",
        s3_secret_key="test-secret-key",
        gravitino_api_base_uri="https://vendor.example/gateway/api/",
    )
    admin = GravitinoCatalogAdmin(config, session=session)

    admin.get_catalog()

    assert session.urls == [
        "https://vendor.example/gateway/api/metalakes/training/catalogs/embeddings"
    ]


def test_catalog_admin_creates_only_the_configured_catalog() -> None:
    session = _CatalogSession(get_status_code=404)
    config = SdkConfig.for_catalog_admin(
        metalake="training",
        catalog_name="embeddings",
        catalog_location="s3://bucket/embeddings",
        s3_endpoint="https://object-store.example",
        s3_region="region-1",
        s3_access_key="test-access-key",
        s3_secret_key="test-secret-key",
        gravitino_api_base_uri="https://vendor.example/gateway/api",
    )

    assert GravitinoCatalogAdmin(config, session=session).ensure_catalog() == "created"
    assert session.created_name == "embeddings"


@pytest.mark.parametrize(
    "api_base_uri",
    (
        "vendor.example/lance/v1",
        "ftp://vendor.example/lance/v1",
        "https://vendor.example/lance/v1?mode=test",
        "https://vendor.example/lance/v1#fragment",
    ),
)
def test_lance_rest_rejects_non_api_base_uris(api_base_uri: str) -> None:
    with pytest.raises(ValueError):
        SdkConfig.for_rest(lance_rest_api_base_uri=api_base_uri)


def test_missing_rest_timeout_raises_instead_of_using_a_default() -> None:
    with pytest.raises(ValueError):
        SdkConfig({}).get_float("REST_TIMEOUT")


def test_lance_rest_describe_returns_canonical_table_handle() -> None:
    session = _RestSession()
    session.request = lambda *args, **kwargs: _Response(
        {
            "location": "s3://bucket/checkpoints/entries/",
            "storage_options": {
                "endpoint": "http://object-store",
                "region": "region-1",
            },
        }
    )
    client = LanceRestCatalog(
        SdkConfig.for_rest(
            lance_rest_api_base_uri="https://vendor.example/gateway/lance"
        ),
        session=session,
    )

    handle = client.describe_table(("catalog", "schema", "entries"))

    assert handle == TableHandle(
        table_id=("catalog", "schema", "entries"),
        location="s3://bucket/checkpoints/entries/",
        storage_options={
            "endpoint": "http://object-store",
            "region": "region-1",
        },
    )


class _RegistrationFailureCatalog:
    def __init__(self) -> None:
        self.registered: list[tuple[str, ...]] = []
        self.dropped: list[tuple[str, ...]] = []

    def describe_table(self, table_id):
        raise LanceRestError(404, "missing")

    def register_table(self, table_id, *, location, properties=None):
        table_id = tuple(table_id)
        self.registered.append(table_id)
        options = {
            "endpoint": "http://object-store",
            "region": "region-1",
            "access_key_id": "access",
            "secret_access_key": "secret",
        }
        if table_id[-1].endswith("page_images"):
            options["region"] = "region-2"
        return TableHandle(table_id, f"{location}/", options)

    def drop_table(self, table_id):
        self.dropped.append(tuple(table_id))

    def close(self):
        return None


def test_gravitino_storage_rolls_back_new_tables_when_initialization_fails() -> None:
    catalog = _RegistrationFailureCatalog()

    with pytest.raises(ValueError, match="different storage options"):
        GravitinoLanceStorage(
            SdkConfig.for_rest(lance_rest_api_base_uri="http://lance/v1"),
            namespace=("catalog", "checkpoint"),
            component_id="embedding",
            location="s3://bucket/checkpoints/embedding",
            catalog=catalog,
        )

    assert catalog.dropped == list(reversed(catalog.registered))


@pytest.mark.parametrize("location", (None, 0, " "))
def test_register_table_rejects_invalid_locations(location: object) -> None:
    client = LanceRestCatalog(
        SdkConfig.for_rest(
            lance_rest_api_base_uri="https://vendor.example/gateway/lance"
        ),
        session=_RestSession(),
    )

    with pytest.raises(ValueError):
        client.register_table(
            ["catalog", "schema", "table"],
            location=location,
        )
