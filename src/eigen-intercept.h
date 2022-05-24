#include "EigenCore.h"
namespace Eigen
{

  namespace internal
  {
    template <
        typename Index,
        typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
        typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
        int ResInnerStride>
    struct general_matrix_matrix_product<Index, LhsScalar, LhsStorageOrder, ConjugateLhs, RhsScalar, RhsStorageOrder, ConjugateRhs, RowMajor, ResInnerStride>;
  }
}