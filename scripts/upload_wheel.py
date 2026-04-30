#!/usr/bin/env python3
"""
SparseEmbedding Release & Deployment Script
-------------------------------------------
A standard release utility for building and publishing the Python package wheel.
Supports multiple publishing targets including PyPI, AWS S3, and MinIO.
"""

import abc
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("release")


class WheelBuilder:
    """Handles the building of the wheel using Bazel."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.bazel_bin_dir = self.project_dir / "bazel-bin"

    def build(self) -> None:
        logger.info("Starting wheel build using Bazel...")
        try:
            subprocess.run(
                ["bazel", "build", "//:sparse_embedding_wheel"],
                cwd=self.project_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Bazel wheel build completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error("Build failed. Stderr output:\n%s", e.stderr)
            raise RuntimeError("Build process encountered an error") from e

    def get_latest_wheel(self) -> Path:
        """Retrieves the most recently modified wheel file in the bazel-bin directory."""
        if not self.bazel_bin_dir.exists():
            raise FileNotFoundError(
                f"Bazel-bin directory not found: {self.bazel_bin_dir}"
            )

        wheels = list(self.bazel_bin_dir.glob("*.whl"))
        if not wheels:
            raise FileNotFoundError(f"No wheels found in {self.bazel_bin_dir}")

        return max(wheels, key=lambda p: p.stat().st_mtime)


class BaseUploader(abc.ABC):
    """Abstract base class defining the standard uploader interface."""

    @abc.abstractmethod
    def upload(self, file_path: Path, dry_run: bool = False) -> None:
        """Uploads the given file to the backend target."""
        pass


class TwineUploader(BaseUploader):
    """Publishes the wheel to PyPI via Twine."""

    def __init__(self, repository: str):
        if not repository:
            raise ValueError("Repository must be provided for TwineUploader")
        self.repository = repository

    def upload(self, file_path: Path, dry_run: bool = False) -> None:
        cmd = [
            sys.executable,
            "-m",
            "twine",
            "upload",
            "--repository",
            self.repository,
            str(file_path),
        ]
        logger.info(
            "Publishing to PyPI repository '%s': %s", self.repository, " ".join(cmd)
        )

        if dry_run:
            logger.info("[Dry Run] Skipped actual upload to PyPI.")
            return

        try:
            subprocess.run(cmd, check=True)
            logger.info("Twine upload successful!")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Twine upload to {self.repository} failed.") from e


class S3Uploader(BaseUploader):
    """Uploads the wheel to an AWS S3 bucket."""

    def __init__(self, s3_uri: str):
        if not s3_uri.startswith("s3://"):
            raise ValueError("S3 URI must start with 's3://'")
        self.s3_uri = s3_uri

    def upload(self, file_path: Path, dry_run: bool = False) -> None:
        cmd = ["aws", "s3", "cp", str(file_path), self.s3_uri]
        logger.info("Copying to S3: %s", " ".join(cmd))

        if dry_run:
            logger.info("[Dry Run] Skipped actual upload to S3.")
            return

        try:
            subprocess.run(cmd, check=True)
            logger.info("S3 upload successful!")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"S3 upload to {self.s3_uri} failed.") from e


class MinioUploader(BaseUploader):
    """Uploads the wheel to a MinIO instance using the 'mc' client."""

    def __init__(self, target_path: str):
        self.target_path = target_path

    def upload(self, file_path: Path, dry_run: bool = False) -> None:
        cmd = ["mc", "cp", str(file_path), self.target_path]
        logger.info("Copying to MinIO: %s", " ".join(cmd))

        if dry_run:
            logger.info("[Dry Run] Skipped actual upload to MinIO.")
            return

        try:
            subprocess.run(cmd, check=True)
            logger.info("MinIO upload successful!")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"MinIO upload to {self.target_path} failed.") from e


class UploaderFactory:
    _uploaders = {
        "pypi": TwineUploader,
        "s3": S3Uploader,
        "minio": MinioUploader,
    }

    @classmethod
    def register(cls, backend: str, uploader_cls: type) -> None:
        cls._uploaders[backend] = uploader_cls

    @classmethod
    def create(cls, backend: str, target: str) -> BaseUploader:
        if backend not in cls._uploaders:
            raise ValueError(f"Unsupported backend: {backend}")
        return cls._uploaders[backend](target)

    @classmethod
    def get_supported_backends(cls) -> list[str]:
        return list(cls._uploaders.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and publish Python package wheels for SparseEmbedding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=UploaderFactory.get_supported_backends(),
        default="pypi",
        help="The deployment backend to use.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help=(
            "Target destination (e.g. 's3://bucket/path/', 'minio1/public/wheels/', "
            "or PyPI repo name). Required for s3 and minio."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the wheel and print the upload command without executing it.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the build step and only attempt to upload the latest existing wheel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target = args.target
    if not target:
        env_vars = {
            "pypi": "SPARSE_EMBEDDING_PYPI_TARGET",
            "minio": "SPARSE_EMBEDDING_MINIO_TARGET",
            "s3": "SPARSE_EMBEDDING_S3_TARGET",
        }
        env_var = env_vars.get(args.backend)
        if env_var and env_var in os.environ:
            target = os.environ[env_var]
        else:
            if args.backend == "pypi":
                target = "pypi"
            else:
                raise ValueError(
                    f"Target destination must be provided via --target or {env_var} environment variable"
                )

    try:
        uploader = UploaderFactory.create(args.backend, target)
    except ValueError as e:
        logger.error("Configuration Error: %s", e)
        sys.exit(1)

    project_dir = Path(__file__).resolve().parent.parent

    try:
        builder = WheelBuilder(project_dir)

        if not args.skip_build:
            builder.build()
        else:
            logger.info("Skipping wheel build step as requested.")

        latest_wheel = builder.get_latest_wheel()
        logger.info("Target wheel identified: %s", latest_wheel.name)

        uploader.upload(latest_wheel, dry_run=args.dry_run)

    except Exception as e:
        logger.error("Deployment aborted: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
