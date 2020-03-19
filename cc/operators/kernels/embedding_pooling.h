#ifndef TENSORFLOW_CORE_KERNELS_FEATUREPOOLING_OP_H_
#define TENSORFLOW_CORE_KERNELS_FEATUREPOOLING_OP_H_

#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"


namespace tensorflow
{
namespace functor
{

template <typename Device, typename T, typename Tidx>
struct LagrangeEmbeddingPoolingFunctor {
    static Status Compute(OpKernelContext *context,
                          int num_shards,
                          bool use_fid_v2,
                          typename TTypes<Tidx, 1>::ConstTensor &instance_ids,
                          typename TTypes<Tidx, 1>::ConstTensor &fids,
                          const std::vector<typename TTypes<T, 2>::ConstTensor> &weight,
                          typename TTypes<Tidx, 1>::ConstTensor &slot_size,
                          typename TTypes<Tidx, 1>::ConstTensor &slot_weight_index,
                          typename TTypes<Tidx, 1>::ConstTensor &slot_output_offset,
                          typename TTypes<Tidx, 1>::ConstTensor &slot_hash_size,
                          typename TTypes<Tidx, 1>::ConstTensor &slot_weight_offset,
                          typename TTypes<T, 2>::Tensor &output);
};

template <typename Device, typename T, typename Tidx>
struct LagrangeEmbeddingUnpoolingFunctor {
    static Status Compute(OpKernelContext *context,
                          bool use_fid_v2,
                          typename TTypes<T, 2>::ConstTensor &output_grads,
                          typename TTypes<Tidx, 1>::ConstTensor &instance_ids,
                          typename TTypes<Tidx, 1>::ConstTensor &fids,
                          typename TTypes<Tidx, 1>::ConstTensor &fid_to_unique_index,
                          typename TTypes<Tidx, 1>::ConstTensor &slot_size,
                          typename TTypes<Tidx, 1>::ConstTensor &slot_weight_index,
                          typename TTypes<Tidx, 1>::ConstTensor &slot_output_offset,
                          std::vector<typename TTypes<T, 2>::Tensor> &output); 
};

} // namespace functor
} // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_FEATUREPOOLING_OP_H_

