"""Fresh-process native restore measurements for two source checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def config(se, threads: int, dimension: int):
    return se.EmbeddingConfig(
        dims=[dimension],
        thread_budget=threads,
        key_dtype="uint64",
        enable_optimizer=False,
        storage_tuning=se.StorageTuningOptions(
            entries_per_page=4096, max_pages=1024, calibrate_page_geometry=False
        ),
    )


def prepare(root: Path, counts: list[int], dimension: int) -> None:
    import lance
    import numpy as np
    import pyarrow as pa

    import sparse_embedding as se
    from sparse_embedding import _core
    from sparse_embedding.checkpoint.schema import ENTRY_SCHEMA

    for count in counts:
        directory = root / str(count)
        directory.mkdir()
        values = {
            "dim": dimension,
            "embedding_values": np.ones(dimension, np.float32).tobytes(),
            "embedding_dtype": "float32",
            "optimizer_state_values": b"",
            "optimizer_state_dtype": "",
            "optimizer_state_stride": 0,
            "step": 0,
            "score": 0,
            "sequence": 2,
            "physical_page_id": 0,
            "page_slot": 0,
            "row_state": "active",
            "source_participant_index": 0,
            "mutation_sequence": 0,
        }
        keys = np.arange(count, dtype=np.uint64) * 2
        arrays = [
            pa.array(
                keys if field.name == "key" else [values[field.name]] * count,
                type=field.type,
            )
            for field in ENTRY_SCHEMA
        ]
        metadata = {
            "sparse_embedding.dimensions": str(dimension),
            "sparse_embedding.embedding_dtype": "float32",
            "sparse_embedding.optimizer_dtype": "",
            "sparse_embedding.score_policy": "none",
            "sparse_embedding.key_dtype": "uint64",
        }
        lance.write_dataset(
            pa.Table.from_arrays(arrays, schema=ENTRY_SCHEMA.with_metadata(metadata)),
            str(directory / "entries.lance"),
        )
        store = se.EmbeddingStore(config(se, 1, dimension))
        store.assign(keys, np.ones((count, dimension), dtype=np.float32))
        fragment = directory / "physical"
        artifact = _core.checkpoint.write_native_fragment(
            store,
            str(fragment),
            0,
            [dimension],
            "float32",
            "float32",
            "none",
            "uint64",
            "restore-benchmark",
            True,
            False,
            "",
            str(fragment / "page_images.lance"),
        )
        (directory / "physical.json").write_text(json.dumps(artifact) + "\n")


def rss_bytes() -> int:
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
        "SC_PAGE_SIZE"
    )


def io_counters() -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (
            line.split(":", 1)
            for line in Path("/proc/self/io").read_text().splitlines()
        )
    }


def task_runtime_ns() -> dict[str, int]:
    result = {}
    for task in Path("/proc/self/task").iterdir():
        try:
            fields = (task / "schedstat").read_text().split()
        except FileNotFoundError:
            continue
        result[task.name] = int(fields[0])
    return result


def measure(args) -> None:
    import numpy as np

    import sparse_embedding as se
    from sparse_embedding import _core

    store = se.EmbeddingStore(config(se, args.threads, args.dimension))
    scratch = Path(os.environ["TMPDIR"])
    baseline_rss = rss_bytes()
    peak_rss = baseline_rss
    peak_scratch = 0
    peak_retired = 0
    samples = 0
    stat_races = 0
    peak_threads = 0
    sampler_tid = None
    stopped = threading.Event()

    def sample() -> None:
        nonlocal \
            peak_rss, \
            peak_scratch, \
            peak_retired, \
            samples, \
            stat_races, \
            peak_threads, \
            sampler_tid
        sampler_tid = str(threading.get_native_id())
        while not stopped.is_set():
            peak_rss = max(peak_rss, rss_bytes())
            peak_retired = max(peak_retired, store.storage_stats()["retired_bytes"])
            size = 0
            for directory, _, names in os.walk(scratch):
                for name in names:
                    try:
                        size += (Path(directory) / name).stat().st_size
                    except FileNotFoundError:
                        stat_races += 1
            peak_scratch = max(peak_scratch, size)
            peak_threads = max(
                peak_threads, len(list(Path("/proc/self/task").iterdir()))
            )
            samples += 1
            stopped.wait(args.sample_interval)

    cpu_before = task_runtime_ns()
    io_before = io_counters()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    sampler_pool = ThreadPoolExecutor(max_workers=1)
    sampler = sampler_pool.submit(sample)
    started = time.perf_counter()
    try:
        directory = Path(args.dataset)
        if args.case == "pages":
            descriptor = json.loads((directory / "physical.json").read_text())
            _core.checkpoint.restore_native_page_images(
                store,
                [str(directory / "physical" / "page_images.lance")],
                0,
                0,
                "uint64",
                {},
                False,
                descriptor["page_layout_generation"],
            )
        else:
            _core.checkpoint.restore_native_partition_entries(
                store,
                [str(directory / "entries.lance")],
                0,
                2 if args.case == "empty" else 1,
                1 if args.case == "empty" else 0,
                "uint64",
                None,
                {},
                False,
            )
    finally:
        elapsed = time.perf_counter() - started
        stopped.set()
        sampler_pool.shutdown(wait=True)
        sampler.result()
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    io_after = io_counters()
    cpu_after = task_runtime_ns()
    final_rss = rss_bytes()
    peak_rss = max(peak_rss, final_rss)
    expected = 0 if args.case == "empty" else args.rows
    if store.total_size() != expected:
        raise AssertionError("restore target row count differs")
    if expected:
        keys = np.array([0, (args.rows - 1) * 2], dtype=np.uint64)
        restored, _ = store.lookup(keys, dim=args.dimension)
        np.testing.assert_array_equal(
            np.from_dlpack(restored).reshape(2, args.dimension),
            np.ones((2, args.dimension), np.float32),
        )
    remaining = list(scratch.iterdir())
    if remaining:
        raise AssertionError(f"native restore left scratch: {remaining}")
    main_tid = str(threading.get_native_id())
    cpu_delta = {
        tid: ticks - cpu_before.get(tid, 0) for tid, ticks in cpu_after.items()
    }
    print(
        json.dumps(
            {
                "checkout": str(Path(args.checkout).resolve()),
                "extension_sha256": hashlib.sha256(
                    Path(_core.__file__).read_bytes()
                ).hexdigest(),
                "case": args.case,
                "source_rows": args.rows,
                "target_rows": expected,
                "thread_budget": args.threads,
                "dimension": args.dimension,
                "seconds": elapsed,
                "initial_rss_bytes": baseline_rss,
                "final_rss_bytes": final_rss,
                "sampled_peak_rss_bytes": peak_rss,
                "rss_increase_bytes": peak_rss - baseline_rss,
                "peak_over_final_rss_bytes": peak_rss - final_rss,
                "sampled_peak_scratch_bytes": peak_scratch,
                "sampled_peak_retired_bytes": peak_retired,
                "final_storage_stats": store.storage_stats(),
                "samples": samples,
                "sample_interval_seconds": args.sample_interval,
                "scratch_stat_races": stat_races,
                "peak_process_threads": peak_threads,
                "background_threads_with_cpu_ns": sum(
                    delta > 0
                    for tid, delta in cpu_delta.items()
                    if tid not in {main_tid, sampler_tid}
                ),
                "preexisting_background_threads_with_cpu_ns": sum(
                    delta > 0
                    for tid, delta in cpu_delta.items()
                    if tid in cpu_before and tid not in {main_tid, sampler_tid}
                ),
                "background_cpu_seconds": sum(
                    delta
                    for tid, delta in cpu_delta.items()
                    if tid not in {main_tid, sampler_tid}
                )
                / 1_000_000_000,
                "user_cpu_seconds": usage_after.ru_utime - usage_before.ru_utime,
                "system_cpu_seconds": usage_after.ru_stime - usage_before.ru_stime,
                "process_io_delta": {
                    key: value - io_before[key] for key, value in io_after.items()
                },
            }
        ),
        flush=True,
    )


def compare(args) -> None:
    root = Path(args.artifacts).resolve()
    root.mkdir()
    datasets = Path(args.datasets).resolve()
    script = Path(__file__).resolve()
    records = []
    for checkout in args.checkouts:
        cases = [
            (count, case, threads)
            for count in args.counts
            for case, threads in (("empty", 1), ("entries", 1), ("entries", 4))
        ]
        cases.extend((count, "pages", 1) for count in args.page_counts)
        for count, case, threads in cases:
            scratch = root / f"scratch-{len(records)}"
            scratch.mkdir()
            environment = os.environ | {
                "PYTHONPATH": str(Path(checkout).resolve()),
                "TMPDIR": str(scratch),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "measure",
                    "--checkout",
                    checkout,
                    "--dataset",
                    str(datasets / str(count)),
                    "--rows",
                    str(count),
                    "--case",
                    case,
                    "--threads",
                    str(threads),
                    "--dimension",
                    str(args.dimension),
                ],
                cwd=checkout,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            (root / f"worker-{len(records)}.log").write_text(result.stderr)
            result.check_returncode()
            record = json.loads(result.stdout)
            records.append(record)
            print(json.dumps(record), flush=True)
            (root / "results.json").write_text(json.dumps(records, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--checkouts", nargs="+", required=True)
    compare_parser.add_argument("--datasets", required=True)
    compare_parser.add_argument("--page-counts", type=int, nargs="*", default=[])
    compare_parser.add_argument("--artifacts", required=True)
    compare_parser.add_argument(
        "--counts", type=int, nargs="+", default=[16384, 65536, 262144]
    )
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--checkout", required=True)
    prepare_parser.add_argument("--artifacts", required=True)
    prepare_parser.add_argument(
        "--counts", type=int, nargs="+", default=[16384, 65536, 262144]
    )
    measure_parser = commands.add_parser("measure")
    measure_parser.add_argument("--checkout", required=True)
    measure_parser.add_argument("--dataset", required=True)
    measure_parser.add_argument("--rows", type=int, required=True)
    measure_parser.add_argument(
        "--case", choices=["empty", "entries", "pages"], required=True
    )
    measure_parser.add_argument("--threads", type=int, required=True)
    measure_parser.add_argument("--sample-interval", type=float, default=0.0005)
    for command in (compare_parser, prepare_parser, measure_parser):
        command.add_argument("--dimension", type=int, default=4)
    args = parser.parse_args()
    if args.command == "measure":
        sys.path.insert(0, args.checkout)
        measure(args)
    elif args.command == "prepare":
        sys.path.insert(0, args.checkout)
        root = Path(args.artifacts).resolve()
        root.mkdir()
        prepare(root, args.counts, args.dimension)
    else:
        compare(args)


if __name__ == "__main__":
    main()
