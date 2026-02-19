#pragma once

#include "zagryadskov_m_complex_spmm_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace zagryadskov_m_complex_spmm_ccs {

class ZagryadskovMComplexSpMMCCSOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }
  explicit ZagryadskovMComplexSpMMCCSOMP(const InType &in);

 private:
  void SpMM(const CCS &a, const CCS &b, CCS &c);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zagryadskov_m_complex_spmm_ccs
