#pragma once

#include <cstddef>

#include "morozova_s_strassen_multiplication/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozova_s_strassen_multiplication {

class MorozovaSStrassenMultiplicationSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MorozovaSStrassenMultiplicationSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] Matrix AddMatrix(const Matrix &a, const Matrix &b) const;
  [[nodiscard]] Matrix SubtractMatrix(const Matrix &a, const Matrix &b) const;
  [[nodiscard]] Matrix MultiplyStrassen(const Matrix &a, const Matrix &b, int leaf_size = 64) const;
  [[nodiscard]] Matrix MultiplyStandard(const Matrix &a, const Matrix &b) const;
  void SplitMatrix(const Matrix &m, Matrix &m11, Matrix &m12, Matrix &m21, Matrix &m22) const;
  [[nodiscard]] Matrix MergeMatrices(const Matrix &m11, const Matrix &m12, const Matrix &m21, const Matrix &m22) const;

  Matrix a_, b_, c_;
  int n_;
};
}  // namespace morozova_s_strassen_multiplication
