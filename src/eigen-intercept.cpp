#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

#include "eigen-intercept.h"

#include <vector>
#include <iostream>

//#include </usr/local/include/eigen3/Eigen/src/Core/products/*>
//_ZN5Eigen8internal29general_matrix_matrix_productIlfLi0ELb0EfLi0ELb0ELi0ELi1EE3runElllPKflS4_lPfllfRNS0_15level3_blockingIffEEPNS0_16GemmParallelInfoIlEE
namespace Eigen
{

  namespace internal
  {
    
    enum StorageOptions
    {
      /** Storage order is column major (see \ref TopicStorageOrders). */
      ColMajor = 0,
      /** Storage order is row major (see \ref TopicStorageOrders). */
      RowMajor = 0x1, // it is only a coincidence that this is equal to RowMajorBit -- don't rely on that
      /** Align the matrix itself if it is vectorizable fixed-size */
      AutoAlign = 0,
      /** Don't require alignment for the matrix itself (the array of coefficients, if dynamically allocated, may still be requested to be aligned) */ // FIXME --- clarify the situation
      DontAlign = 0x2
    };

    
    template <typename _LhsScalar, typename _RhsScalar>
    class level3_blocking;

    /* Specialization for a row-major destination matrix => simple transposition of the product */
    template <
        typename Index,
        typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
        typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
        int ResInnerStride>
    struct general_matrix_matrix_product<Index, LhsScalar, LhsStorageOrder, ConjugateLhs, RhsScalar, RhsStorageOrder, ConjugateRhs, RowMajor, ResInnerStride>
    {
      typedef gebp_traits<RhsScalar, LhsScalar> Traits;

      typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;

      static EIGEN_STRONG_INLINE void run(
          Index rows, Index cols, Index depth,
          const LhsScalar *lhs, Index lhsStride,
          const RhsScalar *rhs, Index rhsStride,
          ResScalar *res, Index resIncr, Index resStride,
          ResScalar alpha,
          level3_blocking<RhsScalar, LhsScalar> &blocking,
          GemmParallelInfo<Index> *info = 0)
      {
        // transpose the product such that the result is column major
        general_matrix_matrix_product<Index,
                                      RhsScalar, RhsStorageOrder == RowMajor ? ColMajor : RowMajor, ConjugateRhs,
                                      LhsScalar, LhsStorageOrder == RowMajor ? ColMajor : RowMajor, ConjugateLhs,
                                      ColMajor, ResInnerStride>::run(cols, rows, depth, rhs, rhsStride, lhs, lhsStride, res, resIncr, resStride, alpha, blocking, info);
      }
    };

    /*  Specialization for a col-major destination matrix
     *    => Blocking algorithm following Goto's paper */
    template <
        typename Index,
        typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
        typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
        int ResInnerStride>
    struct general_matrix_matrix_product<Index, LhsScalar, LhsStorageOrder, ConjugateLhs, RhsScalar, RhsStorageOrder, ConjugateRhs, ColMajor, ResInnerStride>
    {

      typedef gebp_traits<LhsScalar, RhsScalar> Traits;

      typedef typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType ResScalar;
      static void run(Index rows, Index cols, Index depth,
                      const LhsScalar *_lhs, Index lhsStride,
                      const RhsScalar *_rhs, Index rhsStride,
                      ResScalar *_res, Index resIncr, Index resStride,
                      ResScalar alpha,
                      level3_blocking<LhsScalar, RhsScalar> &blocking,
                      GemmParallelInfo<Index> *info = 0)
      {
        std::cout << "TEST" << std::endl;
        void *orig_gemm = dlsym(RTLD_NEXT, "_ZN5Eigen8internal29general_matrix_matrix_productIlfLi0ELb0EfLi0ELb0ELi0ELi1EE3runElllPKflS4_lPfllfRNS0_15level3_blockingIffEEPNS0_16GemmParallelInfoIlEE");
        orig_gemm(rows, cols, depth, _lhs, lhsStride, _rhs, rhsStride, _res, resIncr, resStride, alpha, blocking, info);
        /*
        typedef const_blas_data_mapper<LhsScalar, Index, LhsStorageOrder> LhsMapper;
        typedef const_blas_data_mapper<RhsScalar, Index, RhsStorageOrder> RhsMapper;
        typedef blas_data_mapper<typename Traits::ResScalar, Index, ColMajor, Unaligned, ResInnerStride> ResMapper;
        LhsMapper lhs(_lhs, lhsStride);
        RhsMapper rhs(_rhs, rhsStride);
        ResMapper res(_res, resStride, resIncr);

        Index kc = blocking.kc();                   // cache block size along the K direction
        Index mc = (std::min)(rows, blocking.mc()); // cache block size along the M direction
        Index nc = (std::min)(cols, blocking.nc()); // cache block size along the N direction

        gemm_pack_lhs<LhsScalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing, LhsStorageOrder> pack_lhs;
        gemm_pack_rhs<RhsScalar, Index, RhsMapper, Traits::nr, RhsStorageOrder> pack_rhs;
        gebp_kernel<LhsScalar, RhsScalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;

#ifdef EIGEN_HAS_OPENMP
        if (info)
        {
          // this is the parallel version!
          int tid = omp_get_thread_num();
          int threads = omp_get_num_threads();

          LhsScalar *blockA = blocking.blockA();
          eigen_internal_assert(blockA != 0);

          std::size_t sizeB = kc * nc;
          ei_declare_aligned_stack_constructed_variable(RhsScalar, blockB, sizeB, 0);

          // For each horizontal panel of the rhs, and corresponding vertical panel of the lhs...
          for (Index k = 0; k < depth; k += kc)
          {
            const Index actual_kc = (std::min)(k + kc, depth) - k; // => rows of B', and cols of the A'

            // In order to reduce the chance that a thread has to wait for the other,
            // let's start by packing B'.
            pack_rhs(blockB, rhs.getSubMapper(k, 0), actual_kc, nc);

            // Pack A_k to A' in a parallel fashion:
            // each thread packs the sub block A_k,i to A'_i where i is the thread id.

            // However, before copying to A'_i, we have to make sure that no other thread is still using it,
            // i.e., we test that info[tid].users equals 0.
            // Then, we set info[tid].users to the number of threads to mark that all other threads are going to use it.
            while (info[tid].users != 0)
            {
            }
            info[tid].users = threads;

            pack_lhs(blockA + info[tid].lhs_start * actual_kc, lhs.getSubMapper(info[tid].lhs_start, k), actual_kc, info[tid].lhs_length);

            // Notify the other threads that the part A'_i is ready to go.
            info[tid].sync = k;

            // Computes C_i += A' * B' per A'_i
            for (int shift = 0; shift < threads; ++shift)
            {
              int i = (tid + shift) % threads;

              // At this point we have to make sure that A'_i has been updated by the thread i,
              // we use testAndSetOrdered to mimic a volatile access.
              // However, no need to wait for the B' part which has been updated by the current thread!
              if (shift > 0)
              {
                while (info[i].sync != k)
                {
                }
              }

              gebp(res.getSubMapper(info[i].lhs_start, 0), blockA + info[i].lhs_start * actual_kc, blockB, info[i].lhs_length, actual_kc, nc, alpha);
            }

            // Then keep going as usual with the remaining B'
            for (Index j = nc; j < cols; j += nc)
            {
              const Index actual_nc = (std::min)(j + nc, cols) - j;

              // pack B_k,j to B'
              pack_rhs(blockB, rhs.getSubMapper(k, j), actual_kc, actual_nc);

              // C_j += A' * B'
              gebp(res.getSubMapper(0, j), blockA, blockB, rows, actual_kc, actual_nc, alpha);
            }

            // Release all the sub blocks A'_i of A' for the current thread,
            // i.e., we simply decrement the number of users by 1
            for (Index i = 0; i < threads; ++i)
#if !EIGEN_HAS_CXX11_ATOMIC
#pragma omp atomic
#endif
              info[i].users -= 1;
          }
        }
        else
#endif // EIGEN_HAS_OPENMP
        {
          EIGEN_UNUSED_VARIABLE(info);

          // this is the sequential version!
          std::size_t sizeA = kc * mc;
          std::size_t sizeB = kc * nc;

          ei_declare_aligned_stack_constructed_variable(LhsScalar, blockA, sizeA, blocking.blockA());
          ei_declare_aligned_stack_constructed_variable(RhsScalar, blockB, sizeB, blocking.blockB());

          const bool pack_rhs_once = mc != rows && kc == depth && nc == cols;

          // For each horizontal panel of the rhs, and corresponding panel of the lhs...
          for (Index i2 = 0; i2 < rows; i2 += mc)
          {
            const Index actual_mc = (std::min)(i2 + mc, rows) - i2;

            for (Index k2 = 0; k2 < depth; k2 += kc)
            {
              const Index actual_kc = (std::min)(k2 + kc, depth) - k2;

              // OK, here we have selected one horizontal panel of rhs and one vertical panel of lhs.
              // => Pack lhs's panel into a sequential chunk of memory (L2/L3 caching)
              // Note that this panel will be read as many times as the number of blocks in the rhs's
              // horizontal panel which is, in practice, a very low number.
              pack_lhs(blockA, lhs.getSubMapper(i2, k2), actual_kc, actual_mc);

              // For each kc x nc block of the rhs's horizontal panel...
              for (Index j2 = 0; j2 < cols; j2 += nc)
              {
                const Index actual_nc = (std::min)(j2 + nc, cols) - j2;

                // We pack the rhs's block into a sequential chunk of memory (L2 caching)
                // Note that this block will be read a very high number of times, which is equal to the number of
                // micro horizontal panel of the large rhs's panel (e.g., rows/12 times).
                if ((!pack_rhs_once) || i2 == 0)
                  pack_rhs(blockB, rhs.getSubMapper(k2, j2), actual_kc, actual_nc);

                // Everything is packed, we can now call the panel * block kernel:
                gebp(res.getSubMapper(i2, j2), blockA, blockB, actual_mc, actual_kc, actual_nc, alpha);
              }
            }
          }
        }
        */
      }
    };
  }
}
