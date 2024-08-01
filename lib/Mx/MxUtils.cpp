#include "Mx/MxOps.h"
#include "Mx/MxUtils.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mx;

SmallVector<int64_t> mx::makeShapeLLVMCompatible(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> updatedShape(shape);
  int64_t kDynamic = ShapedType::kDynamic;
  for (unsigned i = 0; i < shape.size(); i++) {
    assert(shape[i] >= 0 || shape[i] == kUnknownSize);
    if (shape[i] == kUnknownSize)
      updatedShape[i] = kDynamic;
  }
  return updatedShape;
}

SmallVector<int64_t> mx::makeShapeMxCompatible(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> updatedShape(shape);
  int64_t kDynamic = ShapedType::kDynamic;
  for (unsigned i = 0; i < shape.size(); i++) {
    assert(shape[i] >= 0 || shape[i] == kDynamic);
    if (shape[i] == kDynamic)
      updatedShape[i] = kUnknownSize;
  }
  return updatedShape;
}

