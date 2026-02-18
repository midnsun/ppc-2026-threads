#pragma once

#include "zagryadskov_m_complex_spmm_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace zagryadskov_m_complex_spmm_ccs {

class ZagryadskovMComplexSpMMCCSSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZagryadskovMComplexSpMMCCSSEQ(const InType &in);

  static void SpMM(const CCS& A, const CCS& B, CCS& C);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zagryadskov_m_complex_spmm_ccs
