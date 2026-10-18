#pragma once

#include <optional>

#include "sparse_embedding/src/checkpoint/external_sort.hpp"

namespace se::checkpoint::test {

inline void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void require_throws(Function &&function, const char *message) {
  try {
    std::forward<Function>(function)();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

class ScratchRoot {
  TemporaryRunDirectory directory_;
  std::optional<std::string> previous_;

public:
  ScratchRoot() {
    if (const auto *value = std::getenv("TMPDIR"))
      previous_ = value;
    (void)directory_.file("root");
    require(::setenv("TMPDIR", directory_.path().c_str(), 1) == 0,
            "cannot configure private test scratch");
  }

  ~ScratchRoot() {
    if (previous_)
      ::setenv("TMPDIR", previous_->c_str(), 1);
    else
      ::unsetenv("TMPDIR");
  }

  const auto &path() const { return directory_.path(); }
};

template <typename Record>
void check_whole_record_truncation(Record first, Record second) {
  ScratchRoot root;
  ExternalSortedRecords<Record, std::less<Record>, 2, 3> records;
  records.append(first);
  records.append(second);
  size_t files = 0;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(root.path())) {
    if (entry.is_regular_file()) {
      std::filesystem::resize_file(entry.path(), sizeof(Record));
      ++files;
    }
  }
  require(files == 1, "truncation fixture did not spill one run");
  size_t visits = 0;
  require_throws<std::runtime_error>(
      [&] {
        records.consume_sorted([&](const Record &) {
          ++visits;
          return true;
        });
      },
      "whole-record truncation was accepted");
  require(visits == 0 && std::filesystem::is_empty(root.path()),
          "truncated run reached its visitor or leaked scratch");
}

} // namespace se::checkpoint::test
