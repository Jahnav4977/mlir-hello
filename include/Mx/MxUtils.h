#ifndef MX_MXUTILS_H
#define MX_MXUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h" // from @llvm-project
#include "mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/Support/LLVM.h" 

using namespace mlir;
namespace mx{

    SmallVector<int64_t> makeShapeLLVMCompatible(ArrayRef<int64_t> shape);
    SmallVector<int64_t> makeShapeMxCompatible(ArrayRef<int64_t> shape);

    constexpr static int64_t kUnknownSize = -1;
}

#endif // MX_MXUTILS_H