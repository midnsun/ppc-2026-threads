#pragma once

#include "zagryadskov_m_complex_spmm_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace zagryadskov_m_complex_spmm_ccs {

class ZagryadskovMComplexSpMMCCSTBB : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kTBB;
  }
  explicit ZagryadskovMComplexSpMMCCSTBB(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zagryadskov_m_complex_spmm_ccs
