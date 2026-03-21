#include "akimov_i_radixsort_int_merge/omp/include/ops_omp.hpp"

#include <omp.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "akimov_i_radixsort_int_merge/common/include/common.hpp"

namespace akimov_i_radixsort_int_merge {

AkimovIRadixSortIntMergeOMP::AkimovIRadixSortIntMergeOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool AkimovIRadixSortIntMergeOMP::ValidationImpl() {
  return !GetInput().empty();
}

bool AkimovIRadixSortIntMergeOMP::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool AkimovIRadixSortIntMergeOMP::RunImpl() {
  auto &arr = GetOutput();
  if (arr.empty()) {
    return true;
  }

  constexpr int32_t kSignMask = INT32_MIN;  // 0x80000000

#pragma omp parallel for default(none) shared(arr, kSignMask)
  for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
    arr[i] ^= kSignMask;
  }

  const int num_bytes = static_cast<int>(sizeof(int));
  std::vector<int> temp(arr.size());

  for (int byte_pos = 0; byte_pos < num_bytes; ++byte_pos) {
    int num_threads = omp_get_max_threads();

    std::vector<std::vector<int>> local_counts(num_threads, std::vector<int>(256, 0));

#pragma omp parallel default(none) shared(arr, byte_pos, local_counts)
    {
      int tid = omp_get_thread_num();
      std::vector<int> &local_count = local_counts[tid];
#pragma omp for
      for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
        uint8_t byte = (arr[i] >> (byte_pos * 8)) & 0xFF;
        ++local_count[byte];
      }
    }

    std::vector<int> count(256, 0);
    for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
      for (int i = 0; i < 256; ++i) {
        count[i] += local_counts[thread_idx][i];
      }
    }

    for (int i = 1; i < 256; ++i) {
      count[i] += count[i - 1];
    }

    for (int i = static_cast<int>(arr.size()) - 1; i >= 0; --i) {
      uint8_t byte = (arr[i] >> (byte_pos * 8)) & 0xFF;
      temp[--count[byte]] = arr[i];
    }

    arr.swap(temp);
  }

#pragma omp parallel for default(none) shared(arr, kSignMask)
  for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
    arr[i] ^= kSignMask;
  }

  return true;
}

bool AkimovIRadixSortIntMergeOMP::PostProcessingImpl() {
  return true;
}

}  // namespace akimov_i_radixsort_int_merge
