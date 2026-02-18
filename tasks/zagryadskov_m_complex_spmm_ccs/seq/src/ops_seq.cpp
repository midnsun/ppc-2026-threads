#include "zagryadskov_m_complex_spmm_ccs/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>
#include <tuple>
#include <complex>
#include <algorithm>

#include "zagryadskov_m_complex_spmm_ccs/common/include/common.hpp"
#include "util/include/util.hpp"

namespace zagryadskov_m_complex_spmm_ccs {

ZagryadskovMComplexSpMMCCSSEQ::ZagryadskovMComplexSpMMCCSSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  CCS emptyOutput;
  GetOutput() = emptyOutput;
}

void ZagryadskovMComplexSpMMCCSSEQ::SpMM(const CCS& A, const CCS& B, CCS& C) {
  C.m = A.m;
  C.n = B.n;
  C.col_ptr.assign(B.n + 1, 0);
  C.row_ind.clear();
  C.values.clear();
  std::complex<double> zero(0.0, 0.0);
  std::vector<int> rows;
  std::vector<int> marker(A.m, -1);
  std::vector<std::complex<double>> acc(A.m);
  const double eps = 1e-14;

  for (int j = 0; j < B.n; ++j) {
    rows.clear();

    for (int k = B.col_ptr[j]; k < B.col_ptr[j + 1]; ++k) {
      std::complex<double> tmpval = B.values[k];
      int Btmpind = B.row_ind[k];

      for (int p = A.col_ptr[Btmpind]; p < A.col_ptr[Btmpind + 1]; ++p) {
        int Atmpind = A.row_ind[p];
        acc[Atmpind] += tmpval*A.values[p];
        if (marker[Atmpind] != j) {
          rows.push_back(Atmpind);
          marker[Atmpind] = j;
        }

      }
    }

    for (size_t i = 0; i < rows.size(); ++i) {
      int tmpind = rows[i];
      if (std::abs(acc[tmpind]) > eps) {
        C.values.push_back(acc[tmpind]);
        C.row_ind.push_back(tmpind);
        ++C.col_ptr[j + 1];
      }
      acc[tmpind] = zero;
    }

    C.col_ptr[j + 1] += C.col_ptr[j];
  }
}

bool ZagryadskovMComplexSpMMCCSSEQ::ValidationImpl() {
  CCS A = std::get<0>(GetInput());
  CCS B = std::get<1>(GetInput());
  return A.n == B.m;
}

bool ZagryadskovMComplexSpMMCCSSEQ::PreProcessingImpl() {
  return true;
}

bool ZagryadskovMComplexSpMMCCSSEQ::RunImpl() {
  
  const CCS& A = std::get<0>(GetInput());
  const CCS& B = std::get<1>(GetInput());
  CCS& C = GetOutput();

  SpMM(A, B, C);

  return true;
}

bool ZagryadskovMComplexSpMMCCSSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zagryadskov_m_complex_spmm_ccs
