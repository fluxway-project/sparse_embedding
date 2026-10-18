#pragma once

#include <cstddef>
#include <cstdint>

namespace se::checkpoint {

struct ExternalSortStatistics {
  uint64_t records = 0;
  uint64_t initial_runs = 0;
  uint64_t merges = 0;
  uint64_t bytes_written = 0;
  uint64_t bytes_read = 0;
  size_t peak_buffer_records = 0;
  size_t peak_merge_fan_in = 0;
  size_t peak_run_files = 0;
  size_t active_run_files = 0;
};

struct PayloadSpoolStatistics {
  uint64_t logical_bytes_written = 0;
  uint64_t logical_bytes_read = 0;
  uint64_t seek_operations = 0;
  uint64_t rows_staged = 0;
  uint64_t rows_selected = 0;
  size_t peak_batch_rows = 0;
  size_t peak_batch_bytes = 0;
};

struct NativeEntryScanStatistics {
  uint64_t sources_scanned = 0;
  uint64_t rows_scanned = 0;
  ExternalSortStatistics identity_sort;
  ExternalSortStatistics selection_sort;
  PayloadSpoolStatistics payload;
};

} // namespace se::checkpoint
