#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>

#include "zagryadskov_m_complex_spmm_ccs/common/include/common.hpp"
#include "zagryadskov_m_complex_spmm_ccs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace zagryadskov_m_complex_spmm_ccs {

class ZagryadskovMRunFuncTestsThreads : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    CCS& A = std::get<0>(input_data_);
    CCS& B = std::get<1>(input_data_);
    CCS& C = test_result;

    if (params == 0) {
      A.m = 2;
      A.n = 3;
      A.col_ptr = {0, 1, 2, 3};
      A.row_ind = {0, 1, 0};
      A.values = {1.0, 2.0, 3.0};

      B.m = 3;
      B.n = 2;
      B.col_ptr = {0, 2, 3};
      B.row_ind = {0, 2, 1};
      B.values = {4.0, 5.0, 6.0};

      C.m = 2;
      C.n = 2;
      C.col_ptr = {0, 1, 2};
      C.row_ind = {0, 1};
      C.values = {19.0, 12.0};
    }

    if (params == 1) {
      A.m = 2;
      A.n = 3;
      A.col_ptr = {0, 1, 2, 4};
      A.row_ind = {0, 1, 0, 1};
      A.values  = {1.0, 3.0, 2.0, 4.0};

      B.m = 3;
      B.n = 2;
      B.col_ptr = {0, 2, 4};
      B.row_ind = {0, 1, 1, 2};
      B.values  = {5.0, 6.0, 7.0, 8.0};

      C.m = 2;
      C.n = 2;
      C.col_ptr = {0, 2, 4};
      C.row_ind = {0, 1, 0, 1};
      C.values  = {5.0, 18.0, 16.0, 53.0};
    }

    if (params == 2) {
      A.m = 3;
      A.n = 3;
      A.col_ptr = {0, 1, 2, 3};
      A.row_ind = {0, 1, 2};
      A.values  = {1.0, 2.0, 3.0};

      B.m = 3;
      B.n = 3;
      B.col_ptr = {0, 1, 2, 3};
      B.row_ind = {2, 0, 2};
      B.values  = {5.0, 4.0, 6.0};

      C.m = 3;
      C.n = 3;
      C.col_ptr = {0, 1, 2, 3};
      C.row_ind = {2, 0, 2};
      C.values  = {15.0, 4.0, 18.0};
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool result = true;
    double eps = 1e-14;

    if (test_result.m != output_data.m) result = false;
    if (test_result.n != output_data.n) result = false;
    if (test_result.col_ptr.size() != output_data.col_ptr.size()) result = false;
    if (test_result.row_ind.size() != output_data.row_ind.size()) result = false;
    if (test_result.values.size() != output_data.values.size()) result = false;
    for (size_t i = 0; i < test_result.col_ptr.size(); ++i)
      if (test_result.col_ptr[i] != output_data.col_ptr[i]) result = false;

    for (int j = 0; j < test_result.n; ++j) {
      std::vector<std::pair<int, std::complex<double>>> test;
      std::vector<std::pair<int, std::complex<double>>> output;
      for (int k = test_result.col_ptr[j]; k < test_result.col_ptr[j + 1]; ++k) {
        test.push_back(std::make_pair(test_result.row_ind[k], test_result.values[k]));
        output.push_back(std::make_pair(output_data.row_ind[k], output_data.values[k]));
      }
      auto cmp = [](const auto& x, const auto& y) { return x.first < y.first; };
      std::sort(test.begin(), test.end(), cmp);
      std::sort(output.begin(), output.end(), cmp);
      
      for (size_t i = 0; i < test.size(); ++i) {
        if (test[i].first != output[i].first) result = false;
        if (std::abs(test[i].second - output[i].second) > eps) result = false;
      }
    }
    
    return result;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType test_result{};
};

namespace {

TEST_P(ZagryadskovMRunFuncTestsThreads, FuncCCSTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {0, 1, 2};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ZagryadskovMComplexSpMMCCSSEQ, InType>(kTestParam, PPC_SETTINGS_zagryadskov_m_complex_spmm_ccs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZagryadskovMRunFuncTestsThreads::PrintFuncTestName<ZagryadskovMRunFuncTestsThreads>;

INSTANTIATE_TEST_SUITE_P(RunFuncCCSTest, ZagryadskovMRunFuncTestsThreads, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zagryadskov_m_complex_spmm_ccs
