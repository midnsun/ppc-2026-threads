#include "zagryadskov_m_complex_spmm_ccs/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <vector>
#include <complex>

#include "zagryadskov_m_complex_spmm_ccs/common/include/common.hpp"
#include "util/include/util.hpp"

namespace zagryadskov_m_complex_spmm_css {

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
                                               std::vector<int> &rows,
                                               std::vector<std::complex<double>> &acc, std::vector<int> &marker,
                                               int j) {
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
    if (std::norm(acc[r_idx]) > eps * eps) {
      c.row_ind[write_ptr] = r_idx;
      c.values[write_ptr] = acc[r_idx];
      ++write_ptr;
    }
    acc[r_idx] = zero;
  }
}

void ZagryadskovMComplexSpMMCCSALL::SpMMNumeric(const CCS &a, const CCS &b, CCS &c, const std::complex<double> &zero,
                                                int jstart, int jend) {
  std::vector<int> marker(a.m, -1);
  std::vector<std::complex<double>> acc(a.m, zero);
  std::vector<int> rows;

  for (int j = jstart; j < jend; ++j) {
    SpMMKernel(a, b, c, zero, eps, rows, acc, marker, j);
  }
}

void ZagryadskovMComplexSpMMCCSALL::SpMM(const CCS &a, const CCS &b, CCS &c) {
  c.m = a.m;
  c.n = b.n;
  int world_size = 0;
  int world_rank = 0;
  int err_code = 0;
  const int num_threads = ppc::util::GetNumThreads();
  std::vector<std::thread> threads(num_threads);

  std::complex<double> zero(0.0, 0.0);
  c.col_ptr.assign(c.n + 1, 0);

  for (int tid = 0; tid < num_threads; ++tid) {
    int jstart = (tid * b.n) / num_threads;
    int jend = ((tid + 1) * b.n) / num_threads;
    threads[tid] = std::thread(SpMMSymbolic, std::cref(a), std::cref(b), std::ref(c.col_ptr), jstart, jend);
  }
  for (auto &th : threads) {
    th.join();
  }

  for (int j = 0; j < c.n; ++j) {
    c.col_ptr[j + 1] += c.col_ptr[j];
  }
  int nnz = c.col_ptr[b.n];
  c.row_ind.resize(nnz);
  c.values.resize(nnz);

  for (int tid = 0; tid < num_threads; ++tid) {
    int jstart = (tid * b.n) / num_threads;
    int jend = ((tid + 1) * b.n) / num_threads;
    threads[tid] =
        std::thread(SpMMNumeric, std::cref(a), std::cref(b), std::ref(c), std::cref(zero), jstart, jend);
  }
  for (auto &th : threads) {
    th.join();
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
  }
  else {
    res = true;
  }
  return res;
}

bool ZagryadskovMComplexSpMMCCSALL::PreProcessingImpl() {
  return true;
}

bool ZagryadskovMComplexSpMMCCSALL::RunImpl() {
  int world_size = 0;
  int world_rank = 0;
  int an = 0;
  int bn = 0;
  int am = 0;
  int bm = 0;
  int anz = 0;
  int bnz = 0;
  int nstart = 0;
  int nend = 0;
  int cn = 0;
  int cm = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  CCS a = CCS();
  CCS b = CCS();
  CCS c = CCS();
  std::vector<int> sencounts_col(world_size);
  std::vector<int> displs_col(world_size);
  std::vector<int> sendcounts_val(world_size);
  std::vector<int> displs_val(world_size);

  if (world_rank == 0) {
    a = std::get<0>(GetInput());
    b = std::get<1>(GetInput());
    an = a.n;
    am = a.m;
    anz = static_cast<int>(a.values.size());
    bn = b.n;
    bm = b.m;
    bnz = static_cast<int>(b.values.size());
  }

  MPI_Bcast(&an, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&am, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&anz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bn, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bm, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  nstart = (world_rank * bn) / world_size;
  nend = ((world_rank + 1) * bn) / world_size;
  sendcounts_col[world_rank] = nend - nstart;
  displs_col[0] = 0;
  for (int rk = 1; rk < world_size; ++rk) {
    displs_col[rk] = displs_col[rk - 1] + sencounts_col[rk - 1];
  }

  b.n = nend - nstart;
  b.m = bm;
  if (world_rank != 0) {
    a.n = an;
    a.m = am;
    a.col_ptr.resize(an + 1);
    a.row_ind.resize(anz);
    a.values.resize(anz);

    b.col_ptr.resize(nend - nstart + 1);
  }

  MPI_Bcast(a.col_ptr.data(), an + 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(a.row_ind.data(), anz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(a.values.data(), anz, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  MPI_Scatterv(b.col_ptr.data(), sendcounts_col.data(), displs_col.data(), MPI_INT, b.col_ptr.data(), sendcounts_col[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
  sendcounts_val[world_rank] = b.col_ptr.back() - b.col_ptr.front();
  displs_val[0] = 0;
  for (int rk = 1; rk < world_size; ++rk) {
    displs_val[rk] = displs_val[rk - 1] + sencounts_val[rk - 1];
  }

  if (world_rank != 0) {
    b.row_ind.resize(sendcounts_val[world_rank]);
    b.values.resize(sendcounts_val[world_rank]);
  }

  MPI_Scatterv(b.row_ind.data(), sendcounts_val.data(), displs_val.data(), MPI_INT, b.row_ind.data(), sendcounts_val[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(b.values.data(), sendcounts_val.data(), displs_val.data(), MPI_C_DOUBLE_COMPLEX, b.values.data(), sendcounts_val[world_rank], MPI_INT, 0, MPI_COMM_WORLD);

  ZagryadskovMComplexSpMMCCSOMP::SpMM(a, b, c);

  MPI_Gatherv(c.col_ptr.data(), sendcounts_col.data, displs_col.data(), MPI_INT, c.col_ptr.data(), sendcounts_col[world_rank], MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    c.m = a.m;
    c.n = b.n;
    for (int j = 0; j <= c.n; ++j)
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

}  // namespace zagryadskov_m_complex_spmm_css
