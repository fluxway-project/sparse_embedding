from __future__ import annotations

import fcntl
import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import lance
import pyarrow as pa


class LanceWorkspace(Protocol):
    def ensure_root(self) -> None: ...

    def uri(self, relative: str) -> str: ...

    def dataset(
        self, relative: str, *, version: int | None = None
    ) -> lance.LanceDataset: ...

    def write_dataset(
        self, table: pa.Table, relative: str, *, mode: str
    ) -> lance.LanceDataset: ...

    def exists(self, relative: str) -> bool: ...

    def create_prefix(self, relative: str) -> None: ...

    def delete_prefix(self, relative: str) -> None: ...

    def child_prefixes(self, relative: str) -> tuple[str, ...]: ...

    def storage_bytes(self) -> int: ...

    def assert_commit_lock(self) -> None: ...

    @contextmanager
    def commit_lock(self) -> Iterator[None]: ...

    def close(self) -> None: ...


class LocalLanceWorkspace:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def ensure_root(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    def uri(self, relative: str) -> str:
        return str(self._root / relative)

    def dataset(self, relative: str, *, version: int | None = None):
        return lance.dataset(self.uri(relative), version=version)

    def write_dataset(self, table: pa.Table, relative: str, *, mode: str):
        return lance.write_dataset(table, self.uri(relative), mode=mode)

    def exists(self, relative: str) -> bool:
        return (self._root / relative).exists()

    def create_prefix(self, relative: str) -> None:
        (self._root / relative).mkdir(parents=True, exist_ok=False)

    def delete_prefix(self, relative: str) -> None:
        path = self._root / relative
        if path.exists():
            shutil.rmtree(path)

    def child_prefixes(self, relative: str) -> tuple[str, ...]:
        path = self._root / relative
        if not path.exists():
            return ()
        return tuple(item.name for item in path.iterdir() if item.is_dir())

    def storage_bytes(self) -> int:
        if not self._root.exists():
            return 0
        return sum(
            path.stat().st_size for path in self._root.rglob("*") if path.is_file()
        )

    def assert_commit_lock(self) -> None:
        return None

    @contextmanager
    def commit_lock(self) -> Iterator[None]:
        descriptor = os.open(self._root, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def close(self) -> None:
        return None


class S3LanceWorkspace:
    def __init__(
        self,
        root_uri: str,
        storage_options: Mapping[str, str],
        *,
        lock_timeout_seconds: float = 60.0,
        lock_ttl_seconds: float = 300.0,
    ) -> None:
        parsed = urlparse(root_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("Gravitino checkpoint location must be an s3:// URI")
        if lock_timeout_seconds <= 0 or lock_ttl_seconds <= 0:
            raise ValueError("lock timeout and TTL must be positive")
        required = ("endpoint", "region", "access_key_id", "secret_access_key")
        missing = tuple(name for name in required if not storage_options.get(name))
        if missing:
            raise ValueError(
                "Gravitino storage options are missing: " + ", ".join(missing)
            )
        import boto3

        self._root_uri = root_uri.rstrip("/")
        self._bucket = parsed.netloc
        self._root_key = parsed.path.lstrip("/").rstrip("/")
        self._storage_options = dict(storage_options)
        self._client = boto3.client(
            "s3",
            endpoint_url=storage_options["endpoint"],
            region_name=storage_options["region"],
            aws_access_key_id=storage_options["access_key_id"],
            aws_secret_access_key=storage_options["secret_access_key"],
        )
        self._lock_timeout_seconds = lock_timeout_seconds
        self._lock_ttl_seconds = lock_ttl_seconds
        self._thread_state = threading.local()

    def ensure_root(self) -> None:
        return None

    def uri(self, relative: str) -> str:
        return f"{self._root_uri}/{relative.strip('/')}"

    def dataset(self, relative: str, *, version: int | None = None):
        return lance.dataset(
            self.uri(relative),
            version=version,
            storage_options=self._storage_options,
        )

    def write_dataset(self, table: pa.Table, relative: str, *, mode: str):
        return lance.write_dataset(
            table,
            self.uri(relative),
            mode=mode,
            storage_options=self._storage_options,
        )

    def exists(self, relative: str) -> bool:
        response = self._client.list_objects_v2(
            Bucket=self._bucket,
            Prefix=self._key(relative, directory=True),
            MaxKeys=1,
        )
        return bool(response.get("KeyCount"))

    def create_prefix(self, relative: str) -> None:
        from botocore.exceptions import ClientError

        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=f"{self._key(relative, directory=True)}.prefix",
                Body=b"",
                IfNoneMatch="*",
            )
        except ClientError as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status in {409, 412}:
                raise FileExistsError(relative) from error
            raise

    def delete_prefix(self, relative: str) -> None:
        prefix = self._key(relative, directory=True)
        while True:
            response = self._client.list_objects_v2(
                Bucket=self._bucket,
                Prefix=prefix,
            )
            objects = response.get("Contents", ())
            if objects:
                self._client.delete_objects(
                    Bucket=self._bucket,
                    Delete={"Objects": [{"Key": item["Key"]} for item in objects]},
                )
            if not response.get("IsTruncated"):
                return

    def child_prefixes(self, relative: str) -> tuple[str, ...]:
        prefix = self._key(relative, directory=True)
        response = self._client.list_objects_v2(
            Bucket=self._bucket,
            Prefix=prefix,
            Delimiter="/",
        )
        result = []
        for item in response.get("CommonPrefixes", ()):
            child = item["Prefix"].removeprefix(prefix).rstrip("/")
            if child:
                result.append(child)
        return tuple(result)

    def storage_bytes(self) -> int:
        total = 0
        prefix = self._key("", directory=True)
        continuation: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if continuation is not None:
                kwargs["ContinuationToken"] = continuation
            response = self._client.list_objects_v2(**kwargs)
            total += sum(int(item["Size"]) for item in response.get("Contents", ()))
            if not response.get("IsTruncated"):
                return total
            continuation = response.get("NextContinuationToken")
            if not isinstance(continuation, str):
                raise RuntimeError("S3 listing omitted its continuation token")

    def assert_commit_lock(self) -> None:
        held = getattr(self._thread_state, "lock", None)
        if held is None:
            raise RuntimeError("S3 checkpoint commit lock is not held")
        owner, key, state = held
        with state["mutex"]:
            renewal_error = state["error"]
            expected_etag = state["etag"]
            if renewal_error is not None:
                raise RuntimeError(
                    "S3 checkpoint commit lock renewal failed"
                ) from renewal_error
            current, current_etag = self._read_lock(key)
            if (
                current.get("owner") != owner
                or current_etag != expected_etag
                or float(current["expires_at"]) <= time.time()
            ):
                raise RuntimeError("S3 checkpoint commit lock was lost")

    @contextmanager
    def commit_lock(self) -> Iterator[None]:
        owner = uuid.uuid4().hex
        key = self._key("_coordination/commit.lock")
        deadline = time.monotonic() + self._lock_timeout_seconds
        etag = self._acquire_lock(key, owner, deadline)
        state: dict[str, Any] = {
            "etag": etag,
            "error": None,
            "mutex": threading.Lock(),
        }
        stopped = threading.Event()
        self._thread_state.lock = (owner, key, state)

        def renew() -> None:
            interval = self._lock_ttl_seconds / 3
            while not stopped.wait(interval):
                try:
                    with state["mutex"]:
                        state["etag"] = self._replace_lock(key, owner, state["etag"])
                except BaseException as error:
                    with state["mutex"]:
                        state["error"] = error
                    return

        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        operation_error: BaseException | None = None
        try:
            yield
        except BaseException as error:
            operation_error = error
            raise
        finally:
            stopped.set()
            thread.join()
            with state["mutex"]:
                renewal_error = state["error"]
                final_etag = state["etag"]
            try:
                self._client.delete_object(
                    Bucket=self._bucket,
                    Key=key,
                    IfMatch=final_etag,
                )
            except BaseException:
                if operation_error is None:
                    raise
            if renewal_error is not None and operation_error is None:
                raise RuntimeError(
                    "S3 checkpoint commit lock renewal failed"
                ) from renewal_error
            del self._thread_state.lock

    def close(self) -> None:
        self._client.close()

    def _acquire_lock(self, key: str, owner: str, deadline: float) -> str:
        from botocore.exceptions import ClientError

        while True:
            payload = self._lock_payload(owner)
            try:
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=payload,
                    IfNoneMatch="*",
                )
                return self._canonical_etag(key)
            except ClientError as error:
                status = error.response.get("ResponseMetadata", {}).get(
                    "HTTPStatusCode"
                )
                if status not in {409, 412}:
                    raise
            current, etag = self._read_lock(key)
            if float(current["expires_at"]) <= time.time():
                try:
                    self._client.put_object(
                        Bucket=self._bucket,
                        Key=key,
                        Body=payload,
                        IfMatch=etag,
                    )
                    return self._canonical_etag(key)
                except ClientError as error:
                    status = error.response.get("ResponseMetadata", {}).get(
                        "HTTPStatusCode"
                    )
                    if status not in {409, 412}:
                        raise
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out acquiring S3 checkpoint commit lock")
            time.sleep(0.05)

    def _replace_lock(self, key: str, owner: str, etag: str) -> str:
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=self._lock_payload(owner),
            IfMatch=etag,
        )
        return self._canonical_etag(key)

    def _read_lock(self, key: str) -> tuple[dict[str, Any], str]:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        payload = json.loads(response["Body"].read())
        if not isinstance(payload, dict) or set(payload) != {"owner", "expires_at"}:
            raise ValueError("S3 checkpoint commit lock is corrupt")
        etag = response.get("ETag")
        if not isinstance(etag, str) or not etag:
            raise RuntimeError("S3 checkpoint commit lock has no ETag")
        return payload, self._normalize_etag(etag)

    def _lock_payload(self, owner: str) -> bytes:
        return json.dumps(
            {"owner": owner, "expires_at": time.time() + self._lock_ttl_seconds},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def _canonical_etag(self, key: str) -> str:
        head = self._client.head_object(Bucket=self._bucket, Key=key)
        etag = head.get("ETag")
        if not isinstance(etag, str) or not etag:
            raise RuntimeError("S3 conditional write returned no ETag")
        return self._normalize_etag(etag)

    @staticmethod
    def _normalize_etag(etag: str) -> str:
        return etag.strip('"')

    def _key(self, relative: str, *, directory: bool = False) -> str:
        suffix = relative.strip("/")
        value = self._root_key if not suffix else f"{self._root_key}/{suffix}"
        return value.rstrip("/") + "/" if directory else value
