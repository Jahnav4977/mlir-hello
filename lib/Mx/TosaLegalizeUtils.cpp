#include "Mx/MxOps.h"
#include "Mx/MxUtils.h"
#include "Mx/TosaLegalizeUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"       // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h" // from @llvm-project

namespace mlir {
namespace tosa {

template <typename T>
std::optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape,
                                    std::optional<Type> dtype) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto width = sizeof(T) * 8;
  if constexpr (std::is_same_v<T, bool>)
    width = 1;

  auto const_type =
      RankedTensorType::get(shape, rewriter.getIntegerType(width));
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);

  if (dtype) {
    return rewriter.createOrFold<tosa::CastOp>(
        op->getLoc(), RankedTensorType::get(shape, *dtype), const_op);
  }
  return const_op.getResult();
}

template <>
std::optional<Value> getConstTensor<float>(PatternRewriter &rewriter,
                                           Operation *op, ArrayRef<float> vec,
                                           ArrayRef<int64_t> shape,
                                           std::optional<Type> dtype) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto const_type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto const_attr = DenseElementsAttr::get(const_type, vec);

  auto const_op =
      rewriter.create<tosa::ConstOp>(op->getLoc(), const_type, const_attr);

  if (dtype) {
    return rewriter.createOrFold<tosa::CastOp>(
        op->getLoc(), RankedTensorType::get(shape, *dtype), const_op);
  }
  return const_op.getResult();
}

template std::optional<Value>
getConstTensor<int32_t>(PatternRewriter &, Operation *, ArrayRef<int32_t> vec,
                        ArrayRef<int64_t> shape, std::optional<Type> dtype);
}
}