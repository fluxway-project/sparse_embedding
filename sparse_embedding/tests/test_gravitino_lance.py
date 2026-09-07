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


def test_gravitino_storage_provisions_manifest_table_only() -> None:
    catalog = _RegistrationFailureCatalog()

    storage = GravitinoLanceStorage(
        SdkConfig.for_rest(lance_rest_api_base_uri="http://lance/v1"),
        namespace=("catalog", "checkpoint"),
        component_id="embedding",
        location="s3://bucket/checkpoints/embedding",
        catalog=catalog,
    )
    storage.close()

    assert catalog.registered == [
        ("catalog", "checkpoint", "embedding_snapshot_manifest")
    ]
    assert catalog.dropped == []


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


def test_s3_lance_workspace_atomic_publish_and_pointer_resolution() -> None:
    from io import BytesIO
    from unittest.mock import MagicMock

    from botocore.exceptions import ClientError

    from sparse_embedding.checkpoint._lance_workspace import S3LanceWorkspace

    s3_store: dict[str, bytes] = {}

    def mock_get_object(Bucket: str, Key: str) -> dict[str, object]:
        if Key in s3_store:
            return {"Body": BytesIO(s3_store[Key])}
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    def mock_put_object(
        Bucket: str, Key: str, Body: bytes | str, **kwargs: object
    ) -> dict[str, object]:
        s3_store[Key] = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        return {}

    def mock_list_objects_v2(
        Bucket: str, Prefix: str, **kwargs: object
    ) -> dict[str, object]:
        matches = [k for k in s3_store if k.startswith(Prefix)]
        contents = [{"Key": k, "Size": len(s3_store[k])} for k in matches]
        return {"Contents": contents, "KeyCount": len(contents), "IsTruncated": False}

    def mock_delete_objects(
        Bucket: str, Delete: dict[str, list[dict[str, str]]]
    ) -> dict[str, object]:
        for obj in Delete["Objects"]:
            s3_store.pop(obj["Key"], None)
        return {}

    def mock_delete_object(Bucket: str, Key: str) -> dict[str, object]:
        s3_store.pop(Key, None)
        return {}

    mock_client = MagicMock()
    mock_client.get_object.side_effect = mock_get_object
    mock_client.put_object.side_effect = mock_put_object
    mock_client.list_objects_v2.side_effect = mock_list_objects_v2
    mock_client.delete_objects.side_effect = mock_delete_objects
    mock_client.delete_object.side_effect = mock_delete_object

    opts = {
        "endpoint": "http://127.0.0.1:9000",
        "region": "us-east-1",
        "access_key_id": "key",
        "secret_access_key": "secret",
    }
    ws = S3LanceWorkspace("s3://test-bucket/root", opts)
    ws._client = mock_client
    reader = S3LanceWorkspace("s3://test-bucket/root", opts)
    reader._client = mock_client

    target = "exports/user_embedding"
    staged = "_staging/compaction-uuid-123"
    mock_put_object("test-bucket", f"root/{staged}/entries/data.lance", b"sample_data")

    # Before publication: target is not visible
    assert not reader.exists(target)

    # Publish via single atomic pointer write
    ws.publish_prefix(staged, target)
    assert reader.exists(target)
    assert reader.uri(f"{target}/entries") == f"s3://test-bucket/root/{staged}/entries"

    # Strict delete_prefix requires missing_ok=True if not exists
    with pytest.raises(FileNotFoundError):
        ws.delete_prefix("nonexistent_path")

    # Deleting published target deletes pointer and staged data
    ws.delete_prefix(target)
    assert not ws.exists(target)
    assert not ws.exists(staged)


def test_s3_lance_workspace_schemes() -> None:
    from unittest.mock import MagicMock

    from sparse_embedding.checkpoint._lance_workspace import S3LanceWorkspace

    opts = {
        "endpoint": "http://127.0.0.1:9000",
        "region": "us-east-1",
        "access_key_id": "key",
        "secret_access_key": "secret",
    }
    mock_client = MagicMock()
    from botocore.exceptions import ClientError

    mock_client.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey"}}, "GetObject"
    )
    ws_s3 = S3LanceWorkspace("s3://test-bucket/root", opts)
    ws_s3._client = mock_client
    ws_s3a = S3LanceWorkspace("s3a://test-bucket/root", opts)
    ws_s3a._client = mock_client
    assert ws_s3.uri("fragment") == "s3://test-bucket/root/fragment"
    assert ws_s3a.uri("fragment") == "s3a://test-bucket/root/fragment"

    with pytest.raises(ValueError, match="s3:// or s3a://"):
        S3LanceWorkspace("file:///local/path", opts)


def test_pointer_failures_are_not_fallbacks(tmp_path):
    from unittest.mock import MagicMock

    from sparse_embedding.checkpoint._lance_workspace import (
        LocalLanceWorkspace,
        S3LanceWorkspace,
    )

    local = tmp_path / "local"
    pointer = local / "published" / ".manifest_pointer.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError):
        LocalLanceWorkspace(local).uri("published/entries")

    options = {
        "endpoint": "http://127.0.0.1:9000",
        "region": "us-east-1",
        "access_key_id": "key",
        "secret_access_key": "secret",
    }
    client = MagicMock()
    client.get_object.side_effect = RuntimeError("network failure")
    remote = S3LanceWorkspace("s3://bucket/root", options)
    remote._client = client
    with pytest.raises(RuntimeError, match="network failure"):
        remote.uri("published/entries")
