// Copyright 2026 The Fluxway Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#pragma once

#ifndef SPARSE_EMBEDDING_BACKEND_HOGWILD_PAGE_COMPACTION_PROFILE_HPP_
#define SPARSE_EMBEDDING_BACKEND_HOGWILD_PAGE_COMPACTION_PROFILE_HPP_

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace se {

#if defined(PAGE_COMPACTION_PROFILE)
struct PageCompactionProfileSnapshot {
  uint64_t enumeration_ns;
  uint64_t reservation_ns;
  uint64_t payload_copy_ns;
  uint64_t staging_ns;
  uint64_t preparation_ns;
  uint64_t publication_ns;
  uint64_t retirement_ns;
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
  uint64_t destination_ready_ns;
  uint64_t destination_ready_calls;
  uint64_t destination_ready_pages;
  uint64_t owner_publish_ns;
  uint64_t owner_publish_calls;
  uint64_t owner_clear_ns;
  uint64_t owner_clear_calls;
  uint64_t locator_publish_ns;
  uint64_t locator_publish_calls;
  uint64_t locator_lookup_ns;
  uint64_t locator_lookup_calls;
  uint64_t owner_validate_ns;
  uint64_t owner_validate_calls;
#endif
#if defined(PAGE_COMPACTION_ADMISSION_PROFILE)
  uint64_t publication_admission_misses;
  uint64_t publication_map_misses;
  uint64_t publication_yields;
#endif
#if defined(PAGE_COMPACTION_SHARD_PROFILE)
  struct ShardPublicationSnapshot {
    uint64_t acquisitions;
    uint64_t compaction_misses;
    uint64_t map_misses;
    uint64_t protected_keys;
    uint64_t hold_ns;
  };
  std::vector<ShardPublicationSnapshot> shard_publication;
#endif
};

struct PageCompactionProfile {
  std::atomic<uint64_t> enumeration_ns{0};
  std::atomic<uint64_t> reservation_ns{0};
  std::atomic<uint64_t> payload_copy_ns{0};
  std::atomic<uint64_t> staging_ns{0};
  std::atomic<uint64_t> preparation_ns{0};
  std::atomic<uint64_t> publication_ns{0};
  std::atomic<uint64_t> retirement_ns{0};
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
  std::atomic<uint64_t> destination_ready_ns{0};
  std::atomic<uint64_t> destination_ready_calls{0};
  std::atomic<uint64_t> destination_ready_pages{0};
  std::atomic<uint64_t> owner_publish_ns{0};
  std::atomic<uint64_t> owner_publish_calls{0};
  std::atomic<uint64_t> owner_clear_ns{0};
  std::atomic<uint64_t> owner_clear_calls{0};
  std::atomic<uint64_t> locator_publish_ns{0};
  std::atomic<uint64_t> locator_publish_calls{0};
  std::atomic<uint64_t> locator_lookup_ns{0};
  std::atomic<uint64_t> locator_lookup_calls{0};
  std::atomic<uint64_t> owner_validate_ns{0};
  std::atomic<uint64_t> owner_validate_calls{0};
#endif
#if defined(PAGE_COMPACTION_ADMISSION_PROFILE)
  std::atomic<uint64_t> publication_admission_misses{0};
  std::atomic<uint64_t> publication_map_misses{0};
  std::atomic<uint64_t> publication_yields{0};
#endif
#if defined(PAGE_COMPACTION_SHARD_PROFILE)
  struct alignas(64) ShardPublicationProfile {
    std::atomic<uint64_t> acquisitions{0};
    std::atomic<uint64_t> compaction_misses{0};
    std::atomic<uint64_t> map_misses{0};
    std::atomic<uint64_t> protected_keys{0};
    std::atomic<uint64_t> hold_ns{0};

    void reset() {
      acquisitions.store(0, std::memory_order_relaxed);
      compaction_misses.store(0, std::memory_order_relaxed);
      map_misses.store(0, std::memory_order_relaxed);
      protected_keys.store(0, std::memory_order_relaxed);
      hold_ns.store(0, std::memory_order_relaxed);
    }
  };

  std::unique_ptr<ShardPublicationProfile[]> shard_publication;
  size_t shard_publication_count = 0;
#endif
};

inline PageCompactionProfile page_compaction_profile;

inline void reset_page_compaction_profile() {
  page_compaction_profile.enumeration_ns.store(0, std::memory_order_relaxed);
  page_compaction_profile.reservation_ns.store(0, std::memory_order_relaxed);
  page_compaction_profile.payload_copy_ns.store(0, std::memory_order_relaxed);
  page_compaction_profile.staging_ns.store(0, std::memory_order_relaxed);
  page_compaction_profile.preparation_ns.store(0, std::memory_order_relaxed);
  page_compaction_profile.publication_ns.store(0, std::memory_order_relaxed);
  page_compaction_profile.retirement_ns.store(0, std::memory_order_relaxed);
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
  page_compaction_profile.destination_ready_ns.store(0,
                                                     std::memory_order_relaxed);
  page_compaction_profile.destination_ready_calls.store(
      0, std::memory_order_relaxed);
  page_compaction_profile.destination_ready_pages.store(
      0, std::memory_order_relaxed);
  page_compaction_profile.owner_publish_ns.store(0, std::memory_order_relaxed);
  page_compaction_profile.owner_publish_calls.store(0,
                                                    std::memory_order_relaxed);
  page_compaction_profile.owner_clear_ns.store(0, std::memory_order_relaxed);
  page_compaction_profile.owner_clear_calls.store(0, std::memory_order_relaxed);
  page_compaction_profile.locator_publish_ns.store(0,
                                                   std::memory_order_relaxed);
  page_compaction_profile.locator_publish_calls.store(
      0, std::memory_order_relaxed);
  page_compaction_profile.locator_lookup_ns.store(0, std::memory_order_relaxed);
  page_compaction_profile.locator_lookup_calls.store(0,
                                                     std::memory_order_relaxed);
  page_compaction_profile.owner_validate_ns.store(0, std::memory_order_relaxed);
  page_compaction_profile.owner_validate_calls.store(0,
                                                     std::memory_order_relaxed);
#endif
#if defined(PAGE_COMPACTION_ADMISSION_PROFILE)
  page_compaction_profile.publication_admission_misses.store(
      0, std::memory_order_relaxed);
  page_compaction_profile.publication_map_misses.store(
      0, std::memory_order_relaxed);
  page_compaction_profile.publication_yields.store(0,
                                                   std::memory_order_relaxed);
#endif
#if defined(PAGE_COMPACTION_SHARD_PROFILE)
  for (size_t shard = 0;
       shard < page_compaction_profile.shard_publication_count; ++shard) {
    page_compaction_profile.shard_publication[shard].reset();
  }
#endif
}

#if defined(PAGE_COMPACTION_SHARD_PROFILE)
inline void configure_page_compaction_shard_profile(size_t shard_count) {
  if (shard_count == 0)
    throw std::invalid_argument("compaction shard profile count is zero");
  if (page_compaction_profile.shard_publication_count == 0) {
    page_compaction_profile.shard_publication =
        std::make_unique<PageCompactionProfile::ShardPublicationProfile[]>(
            shard_count);
    page_compaction_profile.shard_publication_count = shard_count;
    return;
  }
  if (page_compaction_profile.shard_publication_count != shard_count) {
    throw std::logic_error("compaction shard profile count changed");
  }
}

inline void record_page_compaction_shard_acquisition(size_t shard_id,
                                                     size_t protected_keys,
                                                     uint64_t hold_ns) {
  if (shard_id >= page_compaction_profile.shard_publication_count)
    throw std::out_of_range("compaction shard profile id is out of range");
  auto &profile = page_compaction_profile.shard_publication[shard_id];
  profile.acquisitions.fetch_add(1, std::memory_order_relaxed);
  profile.protected_keys.fetch_add(protected_keys, std::memory_order_relaxed);
  profile.hold_ns.fetch_add(hold_ns, std::memory_order_relaxed);
}

inline void record_page_compaction_shard_compaction_miss(size_t shard_id) {
  if (shard_id >= page_compaction_profile.shard_publication_count)
    throw std::out_of_range("compaction shard profile id is out of range");
  page_compaction_profile.shard_publication[shard_id]
      .compaction_misses.fetch_add(1, std::memory_order_relaxed);
}

inline void record_page_compaction_shard_map_miss(size_t shard_id,
                                                  uint64_t hold_ns) {
  if (shard_id >= page_compaction_profile.shard_publication_count)
    throw std::out_of_range("compaction shard profile id is out of range");
  auto &profile = page_compaction_profile.shard_publication[shard_id];
  profile.map_misses.fetch_add(1, std::memory_order_relaxed);
  profile.hold_ns.fetch_add(hold_ns, std::memory_order_relaxed);
}
#endif

inline PageCompactionProfileSnapshot page_compaction_profile_snapshot() {
#if defined(PAGE_COMPACTION_SHARD_PROFILE)
  std::vector<PageCompactionProfileSnapshot::ShardPublicationSnapshot>
      shard_publication;
  shard_publication.reserve(page_compaction_profile.shard_publication_count);
  for (size_t shard = 0;
       shard < page_compaction_profile.shard_publication_count; ++shard) {
    const auto &profile = page_compaction_profile.shard_publication[shard];
    shard_publication.push_back({
        profile.acquisitions.load(std::memory_order_relaxed),
        profile.compaction_misses.load(std::memory_order_relaxed),
        profile.map_misses.load(std::memory_order_relaxed),
        profile.protected_keys.load(std::memory_order_relaxed),
        profile.hold_ns.load(std::memory_order_relaxed),
    });
  }
#endif
  return {
    page_compaction_profile.enumeration_ns.load(std::memory_order_relaxed),
        page_compaction_profile.reservation_ns.load(std::memory_order_relaxed),
        page_compaction_profile.payload_copy_ns.load(std::memory_order_relaxed),
        page_compaction_profile.staging_ns.load(std::memory_order_relaxed),
        page_compaction_profile.preparation_ns.load(std::memory_order_relaxed),
        page_compaction_profile.publication_ns.load(std::memory_order_relaxed),
        page_compaction_profile.retirement_ns.load(std::memory_order_relaxed)
#if defined(PAGE_COMPACTION_DEEP_PROFILE)
            ,
        page_compaction_profile.destination_ready_ns.load(
            std::memory_order_relaxed),
        page_compaction_profile.destination_ready_calls.load(
            std::memory_order_relaxed),
        page_compaction_profile.destination_ready_pages.load(
            std::memory_order_relaxed),
        page_compaction_profile.owner_publish_ns.load(
            std::memory_order_relaxed),
        page_compaction_profile.owner_publish_calls.load(
            std::memory_order_relaxed),
        page_compaction_profile.owner_clear_ns.load(std::memory_order_relaxed),
        page_compaction_profile.owner_clear_calls.load(
            std::memory_order_relaxed),
        page_compaction_profile.locator_publish_ns.load(
            std::memory_order_relaxed),
        page_compaction_profile.locator_publish_calls.load(
            std::memory_order_relaxed),
        page_compaction_profile.locator_lookup_ns.load(
            std::memory_order_relaxed),
        page_compaction_profile.locator_lookup_calls.load(
            std::memory_order_relaxed),
        page_compaction_profile.owner_validate_ns.load(
            std::memory_order_relaxed),
        page_compaction_profile.owner_validate_calls.load(
            std::memory_order_relaxed)
#endif
#if defined(PAGE_COMPACTION_ADMISSION_PROFILE)
            ,
        page_compaction_profile.publication_admission_misses.load(
            std::memory_order_relaxed),
        page_compaction_profile.publication_map_misses.load(
            std::memory_order_relaxed),
        page_compaction_profile.publication_yields.load(
            std::memory_order_relaxed)
#endif
#if defined(PAGE_COMPACTION_SHARD_PROFILE)
            ,
        std::move(shard_publication)
#endif
  };
}
#endif

} // namespace se

#endif // SPARSE_EMBEDDING_BACKEND_HOGWILD_PAGE_COMPACTION_PROFILE_HPP_
