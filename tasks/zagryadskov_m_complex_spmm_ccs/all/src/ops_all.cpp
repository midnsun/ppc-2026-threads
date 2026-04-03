#include "zagryadskov_m_complex_spmm_ccs/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <complex>
#include <vector>

#include "util/include/util.hpp"
#include "zagryadskov_m_complex_spmm_ccs/common/include/common.hpp"

namespace zagryadskov_m_complex_spmm_ccs {

ZagryadskovMComplexSpMMCCSALL::ZagryadskovMComplexSpMMCCSALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int world_rank = 0;
  int err_code = 0;
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  if (world_rank == 0) {
    GetInput() = in;
    GetOutput() = CCS();
  }
}

void ZagryadskovMComplexSpMMCCSALL::SpMMSymbolic(const CCS &a, const CCS &b, std::vector<int> &col_ptr, int jstart,
                                                 int jend) {
  std::vector<int> marker(a.m, -1);

  for (int j = jstart; j < jend; ++j) {
    int count = 0;

    for (int k = b.col_ptr[j]; k < b.col_ptr[j + 1]; ++k) {
      int b_row = b.row_ind[k];
      for (int zp = a.col_ptr[b_row]; zp < a.col_ptr[b_row + 1]; ++zp) {
        int a_row = a.row_ind[zp];
        if (marker[a_row] != j) {
          marker[a_row] = j;
          ++count;
        }
      }
    }
    col_ptr[j + 1] += count;
  }
}

void ZagryadskovMComplexSpMMCCSALL::SpMMKernel(const CCS &a, const CCS &b, CCS &c, const std::complex<double> &zero,
                                               std::vector<int> &rows, std::vector<std::complex<double>> &acc,
                                               std::vector<int> &marker, int j) {
  rows.clear();
  int write_ptr = c.col_ptr[j];

  for (int k = b.col_ptr[j]; k < b.col_ptr[j + 1]; ++k) {
    std::complex<double> tmpval = b.values[k];
    int b_row = b.row_ind[k];
    for (int zp = a.col_ptr[b_row]; zp < a.col_ptr[b_row + 1]; ++zp) {
      int a_row = a.row_ind[zp];
      acc[a_row] += tmpval * a.values[zp];
      if (marker[a_row] != j) {
        marker[a_row] = j;
        rows.push_back(a_row);
      }
    }
  }

  for (int r_idx : rows) {
    c.row_ind[write_ptr] = r_idx;
    c.values[write_ptr] = acc[r_idx];
    ++write_ptr;
    acc[r_idx] = zero;
  }
}

void ZagryadskovMComplexSpMMCCSALL::SpMMNumeric(const CCS &a, const CCS &b, CCS &c, const std::complex<double> &zero,
                                                int jstart, int jend) {
  std::vector<int> marker(a.m, -1);
  std::vector<std::complex<double>> acc(a.m, zero);
  std::vector<int> rows;

  for (int j = jstart; j < jend; ++j) {
    SpMMKernel(a, b, c, zero, rows, acc, marker, j);
  }
}

void ZagryadskovMComplexSpMMCCSALL::SpMM(const CCS &a, const CCS &b, CCS &c) {
  c.m = a.m;
  c.n = b.n;
  const int num_threads = ppc::util::GetNumThreads();

  std::complex<double> zero(0.0, 0.0);
  c.col_ptr.assign(c.n + 1, 0);

#pragma omp parallel default(none) shared(num_threads, a, b, c) num_threads(ppc::util::GetNumThreads())
  {
    int tid = omp_get_thread_num();
    int jstart = (tid * b.n) / num_threads;
    int jend = ((tid + 1) * b.n) / num_threads;
    SpMMSymbolic(a, b, c.col_ptr, jstart, jend);
  }

  for (int j = 0; j < c.n; ++j) {
    c.col_ptr[j + 1] += c.col_ptr[j];
  }
  int nnz = c.col_ptr[b.n];
  c.row_ind.resize(nnz);
  c.values.resize(nnz);
#pragma omp parallel default(none) shared(num_threads, a, b, c, zero) num_threads(ppc::util::GetNumThreads())
  {
    int tid = omp_get_thread_num();
    int jstart = (tid * b.n) / num_threads;
    int jend = ((tid + 1) * b.n) / num_threads;
    SpMMNumeric(a, b, c, zero, jstart, jend);
  }
}

bool ZagryadskovMComplexSpMMCCSALL::ValidationImpl() {
  bool res = false;
  int world_rank = 0;
  int err_code = 0;
  err_code = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (err_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_rank failed");
  }
  if (world_rank == 0) {
    const CCS &a = std::get<0>(GetInput());
    const CCS &b = std::get<1>(GetInput());
    res = a.n == b.m;
  } else {
    res = true;
  }
  return res;
}

bool ZagryadskovMComplexSpMMCCSALL::PreProcessingImpl() {
  return true;
}

bool ZagryadskovMComplexSpMMCCSALL::RunImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    const CCS &a = std::get<0>(GetInput());
    const CCS &b = std::get<1>(GetInput());
    CCS &c = GetOutput();

    ZagryadskovMComplexSpMMCCSALL::SpMM(a, b, c);
  }

  return true;
}

bool ZagryadskovMComplexSpMMCCSALL::PostProcessingImpl() {
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int m;
  int n;
  int nz;
  CCS &c = GetOutput();
  if (world_rank == 0) {
    m = c.m;
    n = c.n;
    nz = static_cast<int>(c.values.size());
  }
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (world_rank != 0) {
    c.m = m;
    c.n = n;
    c.col_ptr.resize(n + 1);
    c.row_ind.resize(nz);
    c.values.resize(nz);
  }
  MPI_Bcast(c.col_ptr.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(c.row_ind.data(), nz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(c.values.data(), nz, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

  return true;
}

}  // namespace zagryadskov_m_complex_spmm_ccs
