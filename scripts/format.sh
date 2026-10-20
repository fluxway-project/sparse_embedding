#!/bin/bash
set -euo pipefail

case "${1:---fix}" in
  --fix) mode=fix ;;
  --check) mode=check ;;
  *) printf 'Usage: bash scripts/format.sh [--fix|--check]\n' >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  printf 'Usage: bash scripts/format.sh [--fix|--check]\n' >&2
  exit 2
fi

if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]]; then
  cd -- "$BUILD_WORKSPACE_DIRECTORY"
fi
workspace="$(git rev-parse --show-toplevel)"
cd -- "$workspace"
for tool in clang-format-16 buildifier ruff; do
  command -v "$tool" > /dev/null
done

mkdir -p tmp/format
file_list="$(mktemp "$PWD/tmp/format/files.XXXXXX")"
trap 'rm -f -- "$file_list"' EXIT
git ls-files --cached --others --exclude-standard -z > "$file_list"

cpp=()
bazel=()
python=()
while IFS= read -r -d '' file; do
  if [[ ! -f "$file" || -L "$file" ]]; then
    continue
  fi
  case "$file" in
    *.h|*.c|*.cc|*.cpp|*.hpp|*.cu|*.cuh|*.inl) cpp+=("$file") ;;
    *.bazel|*.bzl|BUILD|*/BUILD|WORKSPACE|*/WORKSPACE) bazel+=("$file") ;;
    *.py) python+=("$file") ;;
  esac
done < "$file_list"

printf 'Formatting mode: %s; C++: %s, Bazel: %s, Python: %s files\n' \
  "$mode" "${#cpp[@]}" "${#bazel[@]}" "${#python[@]}"
if (( ${#cpp[@]} )); then
  if [[ "$mode" == check ]]; then
    clang-format-16 -style=file --dry-run --Werror "${cpp[@]}"
  else
    clang-format-16 -style=file -i "${cpp[@]}"
  fi
fi
if (( ${#bazel[@]} )); then
  buildifier "-mode=$mode" "${bazel[@]}"
fi
if (( ${#python[@]} )); then
  if [[ "$mode" == check ]]; then
    ruff check --select I "${python[@]}"
    ruff format --check "${python[@]}"
  else
    ruff check --select I --fix "${python[@]}"
    ruff format "${python[@]}"
  fi
fi
