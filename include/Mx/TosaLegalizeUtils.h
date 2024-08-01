#ifndef MX_TOSALEGALIZEUTILS_H
#define MX_TOSALEGALIZEUTILS_H

#include "mlir/Dialect/Quant/QuantTypes.h"        // from @llvm-project
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"   // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"            // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                 // from @llvm-project
#include "mlir/IR/PatternMatch.h"                 // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h" // from @llvm-project
#include "mlir/Support/LLVM.h"                  

namespace mlir{
namespace tosa{
    template <typename T>
    std::optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape,
                                    std::optional<Type> dtype = {});
}
}

#endif // MX_TOSALEGALIZEUTILS_H