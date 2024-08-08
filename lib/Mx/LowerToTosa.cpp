#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "Mx/MxDialect.h"
#include "Mx/MxOps.h"
#include "Mx/MxPasses.h"
#include "Mx/MxUtils.h"
#include "Mx/TosaLegalizeUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <iostream>

using namespace mlir;

namespace{
    
class AddOpLowering : public OpRewritePattern<mx::AddOp>{
    using OpRewritePattern<mx::AddOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mx::AddOp op,PatternRewriter &rewriter) const final{
        auto lhs=op.getLhs();
        auto rhs=op.getRhs();
        auto output=op.getResult();
        auto outputType=llvm::dyn_cast<RankedTensorType>(output.getType());
        rewriter.replaceOpWithNewOp<mlir::tosa::AddOp>(op,outputType,lhs,rhs);
    return success();
    }
};

class MulOpLowering : public OpRewritePattern<mx::MulOp>{
    using OpRewritePattern<mx::MulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mx::MulOp op,PatternRewriter &rewriter) const final{
        auto lhs=op.getLhs();
        auto rhs=op.getRhs();
        auto output=op.getResult();
        auto outputType=llvm::dyn_cast<RankedTensorType>(output.getType());
        rewriter.replaceOpWithNewOp<mlir::tosa::MulOp>(op,outputType,lhs,rhs,0);
    return success();
    }
};

class AddMulOpLowering : public OpRewritePattern<mx::AddMulOp>{
   using OpRewritePattern<mx::AddMulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mx::AddMulOp op,PatternRewriter &rewriter) const final{
       auto first=op.getFirst();
       auto second=op.getSecond();
       auto third=op.getThird();
       auto output=op.getResult();
       auto outputType=llvm::dyn_cast<RankedTensorType>(output.getType());
       Value temp;
       temp = rewriter.create<mx::AddOp>(op->getLoc(), outputType, first,second).getResult();
       rewriter.replaceOpWithNewOp<mx::MulOp>(op,outputType,temp,third);
    return success();
    }
 };
}

class ReshapeOpLowering : public OpRewritePattern<mx::ReshapeOp>{
  using OpRewritePattern<mx::ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mx::ReshapeOp op, PatternRewriter &rewriter) const final{
    auto input = op.getInput1();
    auto new_shape = op.getShape();
    auto output = op.getResult();
    auto outputType=llvm::dyn_cast<RankedTensorType>(output.getType());
    rewriter.replaceOpWithNewOp<mlir::tosa::ReshapeOp>(op,outputType,input,new_shape);
  return success();
  }
};

class TransposeOpLowering : public OpRewritePattern<mx::TransposeOp>{
  using OpRewritePattern<mx::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mx::TransposeOp op, PatternRewriter &rewriter) const final{
    auto input = op.getInput1();
    auto perms = op.getPerms();
    auto output = op.getResult();
    auto outputType=llvm::dyn_cast<RankedTensorType>(output.getType());
    rewriter.replaceOpWithNewOp<mlir::tosa::TransposeOp>(op,outputType,input,perms);
  return success();
  }
};

class TanhOpLowering : public OpRewritePattern<mx::TanhOp>{
  using OpRewritePattern<mx::TanhOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mx::TanhOp op, PatternRewriter &rewriter) const final{
    auto input = op.getInput();
    auto output = op.getResult();
    auto outputType=llvm::dyn_cast<RankedTensorType>(output.getType());
    rewriter.replaceOpWithNewOp<mlir::tosa::TanhOp>(op,outputType,input);
  return success();
  }
};

class Conv2dOpLowering : public OpRewritePattern<mx::Conv2dOp>{
  using OpRewritePattern<mx::Conv2dOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mx::Conv2dOp op, PatternRewriter &rewriter) const final{
    auto input = op.getInput();
    auto weight = op.getWeight();
    auto output = op.getResult();
    auto outputType = llvm::dyn_cast<RankedTensorType>(output.getType());

    auto inputTy = llvm::dyn_cast<RankedTensorType>(input.getType());
    auto weightTy = llvm::dyn_cast<RankedTensorType>(weight.getType());
    auto inputElemTy = inputTy.getElementType();
    auto weightElemTy = weightTy.getElementType();
    auto inputShape = mx::makeShapeMxCompatible(inputTy.getShape());
    auto weightShape = mx::makeShapeMxCompatible(weightTy.getShape());
    auto bias = op.getBias();
    if (isa<mlir::NoneType>(op.getBias().getType())) {
      //SmallVector<float> zeroVec(weightShape[0], 0);
      //bias=std::optional<Value> temp = mlir::tosa::getConstTensor<float>(rewriter, op, zeroVec,
      //                                   {static_cast<int32_t>(weightShape[0])}).value();
      //bias=temp;
      //std::optional<Value> temp = mlir::tosa::getConstTensor<float>(rewriter, op, zeroVec,
      //                                   {static_cast<int32_t>(weightShape[0])}).value();
      //bias=temp;
    }
    auto biasElemTy =
      isa<mlir::FloatType>(inputElemTy) ? inputElemTy : rewriter.getI32Type();

    SmallVector<int64_t> stride({1,1});

    SmallVector<int64_t> padding(
        {0, 0, 0, 0});

    SmallVector<int64_t> dilation({1,1});

    std::optional<Value> nchwToNhwcTransposeConst =
      tosa::getConstTensor<int32_t>(rewriter, op,
                                    /*vec=*/{0, 2, 3, 1},
                                    /*shape=*/{static_cast<int32_t>(4)});
    SmallVector<int64_t> transposedInputShape(
      {inputShape[0], inputShape[2], inputShape[3], inputShape[1]});
    auto transposedInputType = mlir::RankedTensorType::get(
      mx::makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
    auto transposedInput =
      rewriter
          .create<mlir::tosa::TransposeOp>(
              op->getLoc(),
              llvm::dyn_cast<RankedTensorType>(transposedInputType), input,
              nchwToNhwcTransposeConst.value())
          .getResult();
    
    SmallVector<int64_t> transformedWeightShape;
    RankedTensorType transformedWeightType;
    Value transformedWeight;

    int64_t outputCDim;
    transformedWeightShape = {weightShape[0], weightShape[2], weightShape[3],
                              weightShape[1]};
    transformedWeightType = mlir::RankedTensorType::get(
        mx::makeShapeLLVMCompatible(transformedWeightShape), weightElemTy);
    transformedWeight =
        rewriter
            .create<mlir::tosa::TransposeOp>(
                op->getLoc(),
                llvm::dyn_cast<RankedTensorType>(transformedWeightType),weight,
                nchwToNhwcTransposeConst.value())
            .getResult();
    outputCDim = transformedWeightShape[0];

    int64_t outputHDim, outputWDim;
    if (inputTy.hasStaticShape()) {
      int64_t inputHDim = inputShape[2];
      int64_t inputWDim = inputShape[3];
      int64_t weightHDim = weightShape[2];
      int64_t weightWDim = weightShape[3];
      outputHDim = (inputHDim + padding[0] + padding[1] -
                    dilation[0] * (weightHDim - 1) - 1) /
                      stride[0] +
                  1;
      outputWDim = (inputWDim + padding[2] + padding[3] -
                    dilation[1] * (weightWDim - 1) - 1) /
                      stride[1] +
                  1;
    } else {
      outputHDim = mx::kUnknownSize;
      outputWDim = mx::kUnknownSize;
    }
    SmallVector<int64_t> outputShape = {transposedInputShape[0], outputHDim,
                                      outputWDim, outputCDim};

    auto convOpTy =
      mlir::RankedTensorType::get(mx::makeShapeLLVMCompatible(outputShape), biasElemTy);

    Value convOpResult;
    convOpResult =
        rewriter
            .create<mlir::tosa::Conv2DOp>(op->getLoc(),
                                    llvm::dyn_cast<RankedTensorType>(convOpTy),
                                    transposedInput, transformedWeight, bias,
                                    rewriter.getDenseI64ArrayAttr(padding),
                                    rewriter.getDenseI64ArrayAttr(stride),
                                    rewriter.getDenseI64ArrayAttr(dilation))
            .getResult();
    std::optional<Value> nhwcToNchwTransposeConst =
      tosa::getConstTensor<int32_t>(rewriter, op,
                                    /*vec=*/{0, 3, 1, 2},
                                    /*shape=*/{static_cast<int32_t>(4)});
    SmallVector<int64_t> transposedOutputShape(
        {outputShape[0], outputShape[3], outputShape[1], outputShape[2]});
    auto transposedOutputType = mlir::RankedTensorType::get(
        mx::makeShapeLLVMCompatible(transposedOutputShape), biasElemTy);
    auto transposedOutput =
        rewriter
            .create<mlir::tosa::TransposeOp>(
                op->getLoc(),
                llvm::dyn_cast<RankedTensorType>(transposedOutputType),
                convOpResult, nhwcToNchwTransposeConst.value())
            .getResult();

    Value rescaledResult = transposedOutput;
    rewriter.replaceOpWithNewOp<mlir::tosa::CastOp>(
        op, outputType, rescaledResult);

    return success();
  }
};

template <typename OpT, typename TosaOpT>
class ConvertPoolingBaseOp : public OpConversionPattern<OpT> {
public:
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpAdaptor = typename OpT::Adaptor;

  virtual LogicalResult processInputs(OpT op, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter,
                                      Value &input, DenseI64ArrayAttr &kernel,
                                      DenseI64ArrayAttr &stride,
                                      DenseI64ArrayAttr &pad,
                                      Type &outputTy) const {
    return rewriter.notifyMatchFailure(
        op, "Unimplemented pooling input parsing function");
  }

  static int64_t getOutputDim(int64_t inputDim, int64_t kernelDim,
                              int64_t stride, int64_t padBefore,
                              int64_t padAfter) {
    if (inputDim == mx::kUnknownSize) {
      return mx::kUnknownSize;
    } else {
      int64_t dimSize =
          inputDim + padBefore + padAfter - 1 * (kernelDim - 1) - 1;
      return dimSize / stride + 1;
    }
  }

  Value transposeTensor(OpT op, ConversionPatternRewriter &rewriter,
                        Value input, ArrayRef<int32_t> transposeDims) const {
    auto inputTy = llvm::dyn_cast<RankedTensorType>(input.getType());
    auto inputElemTy = inputTy.getElementType();
    auto inputShape = mx::makeShapeMxCompatible(inputTy.getShape());
    auto inputRank = inputTy.getRank();

    std::optional<Value> transposeDimsConst = tosa::getConstTensor<int32_t>(
        rewriter, op,
        /*vec=*/transposeDims,
        /*shape=*/{static_cast<int32_t>(inputRank)});

    SmallVector<int64_t> transposedInputShape;
    for (auto &dim : transposeDims)
      transposedInputShape.push_back(inputShape[dim]);
    auto transposedInputType = mlir::RankedTensorType::get(
        mx::makeShapeLLVMCompatible(transposedInputShape), inputElemTy);
    return rewriter
        .create<tosa::TransposeOp>(op->getLoc(), transposedInputType, input,
                                   transposeDimsConst.value())
        .getResult();
  }

  Value transposePoolingInputToHwc(OpT op,
                                   ConversionPatternRewriter &rewriter,
                                   Value input) const {
    auto inputRank = llvm::dyn_cast<RankedTensorType>(input.getType()).getRank();

    SmallVector<int32_t> nchwToNhwc4DTransposeDims({0, 2, 3, 1});
    SmallVector<int32_t> chwToHwc3DTransposeDims({1, 2, 0});

    return transposeTensor(op, rewriter, input,
                           inputRank == 3 ? chwToHwc3DTransposeDims
                                          : nchwToNhwc4DTransposeDims);
  }

  Value transposePoolingOutputToChw(OpT op,
                                    ConversionPatternRewriter &rewriter,
                                    Value input) const {
    auto inputTy = llvm::dyn_cast<RankedTensorType>(input.getType());
    auto inputRank = inputTy.getRank();

    SmallVector<int32_t> nhwcToNchw4DTransposeDims({0, 3, 1, 2});
    SmallVector<int32_t> hwcToChw3DTransposeDims({2, 0, 1});

    return transposeTensor(op, rewriter, input,
                           inputRank == 3 ? hwcToChw3DTransposeDims
                                          : nhwcToNchw4DTransposeDims);
  }

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input;
    DenseI64ArrayAttr kernel, stride, pad;
    Type outputTy;

    if (failed(processInputs(op, adaptor, rewriter, input, kernel, stride, pad,
                             outputTy)))
      return rewriter.notifyMatchFailure(
          op, "Failed to process inputs for pooling");

    Value pooledOutput;
    pooledOutput = rewriter
                         .create<TosaOpT>(op->getLoc(), outputTy, input, kernel,
                                          stride, pad)
                         .getResult();

    auto transposedOutput =
        ConvertPoolingBaseOp<OpT, TosaOpT>::transposePoolingOutputToChw(
            op, rewriter, pooledOutput);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        llvm::dyn_cast<RankedTensorType>(op.getType()),
        transposedOutput);

    return success();

  }
};

template <typename AtenOpT, typename tosaOp>
static Type getOutputTypeForNonAdaptivePoolingOp(
    RankedTensorType inputTy, SmallVectorImpl<int64_t> &kernelSize,
    SmallVectorImpl<int64_t> &strideArray, SmallVectorImpl<int64_t> &padArray) {
  auto inputShape = mx::makeShapeMxCompatible(inputTy.getShape());
  auto inputRank = inputTy.getRank();
  auto inputElemTy = inputTy.getElementType();

  int64_t outputHDim = ConvertPoolingBaseOp<AtenOpT, tosaOp>::getOutputDim(
      inputShape[inputRank - 2], kernelSize[0], strideArray[0], padArray[0],
      padArray[0]);
  int64_t outputWDim = ConvertPoolingBaseOp<AtenOpT, tosaOp>::getOutputDim(
      inputShape[inputRank - 1], kernelSize[1], strideArray[1], padArray[1],
      padArray[1]);
  padArray[0] = (outputHDim - 1) * strideArray[0] +
                1 * kernelSize[0] - 1 + 1 -
                padArray[0] * 2 - inputShape[inputRank - 2];
  padArray[1] = (outputWDim - 1) * strideArray[1] +
                1 * kernelSize[1] - 1 + 1 -
                padArray[1] * 2 - inputShape[inputRank - 1];
  SmallVector<int64_t> outputShape;
  if (inputRank > 3)
    outputShape.push_back(inputShape[0]);
  outputShape.push_back(outputHDim);
  outputShape.push_back(outputWDim);
  outputShape.push_back(inputShape[inputRank - 3]);
  return mlir::RankedTensorType::get(mx::makeShapeLLVMCompatible(outputShape),
                               inputElemTy);
}

// Checks the validity of pooling parameters and stores them in the respective
// vector. Also, gets the output type for the pooling op.
template <typename AtenOpT, typename tosaOp>
static LogicalResult getOutputTypeAndPoolingParameters(
    AtenOpT op, ConversionPatternRewriter &rewriter, Value inputXchw, Type &outputTy,
    DenseI64ArrayAttr &kernel, DenseI64ArrayAttr &stride,
    DenseI64ArrayAttr &pad) {

  RankedTensorType inputTy = llvm::dyn_cast<RankedTensorType>(inputXchw.getType());
  if (!inputTy)
    return rewriter.notifyMatchFailure(
        op, "Pooling op requires ranked tensor input");

  auto inputRank = inputTy.getRank();
  // Rank sanity check.
  if (inputTy.getRank() != 4 && inputRank != 3)
    return rewriter.notifyMatchFailure(
        op, "NCHW->NHWC transpose requires 3D or 4D tensor");

  SmallVector<int64_t, 2> kernelSizeInts(op.getKernel().begin(),op.getKernel().end());
  SmallVector<int64_t, 2> strideInts(op.getStride().begin(),op.getStride().end());
  SmallVector<int64_t> paddingInts({0,0});
  SmallVector<int64_t> padArr({0,0,0,0});

  kernel = rewriter.getDenseI64ArrayAttr(kernelSizeInts);
  stride = rewriter.getDenseI64ArrayAttr(strideInts);

  outputTy = getOutputTypeForNonAdaptivePoolingOp<AtenOpT, tosaOp>(
      inputTy, kernelSizeInts, strideInts, paddingInts);
  padArr[1] = padArr[1] + paddingInts[0];
  padArr[3] = padArr[3] + paddingInts[1];
  pad = rewriter.getDenseI64ArrayAttr(
      {padArr[0], padArr[1], padArr[2], padArr[3]});
  return success();
}

class MaxPool2dOpLowering
    : public ConvertPoolingBaseOp<mx::MaxPool2dOp, tosa::MaxPool2dOp> {
public:
  using ConvertPoolingBaseOp<mx::MaxPool2dOp,
                                 tosa::MaxPool2dOp>::ConvertPoolingBaseOp;
  LogicalResult processInputs(mx::MaxPool2dOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &input,
                              DenseI64ArrayAttr &kernel,
                              DenseI64ArrayAttr &stride, DenseI64ArrayAttr &pad,
                              Type &outputTy) const override {
    // TOSA pooling only supports unit dilation.
    if (failed(getOutputTypeAndPoolingParameters<mx::MaxPool2dOp,
                                                 tosa::MaxPool2dOp>(
            op, rewriter, adaptor.getInput(), outputTy, kernel,
            stride, pad)))
      return rewriter.notifyMatchFailure(
          op, "invalid pooling parameters or input type");

    // Transpose to xHWC
    input = ConvertPoolingBaseOp<mx::MaxPool2dOp, tosa::MaxPool2dOp>::
        transposePoolingInputToHwc(op, rewriter, adaptor.getInput());

    return success();
  }
};

template<typename OpT>
class ConvertMatmulBaseOp : public OpConversionPattern<OpT> {
public:
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpAdaptor = typename OpT::Adaptor;

  virtual LogicalResult readMatMulInputs(OpT op, OpAdaptor adaptor,
                                         ConversionPatternRewriter &rewriter,
                                         Value &lhs, Value &rhs) const {
    return rewriter.notifyMatchFailure(
        op,
        "Unimplemented matrix multiplication variant input parsing function");
  }

  LogicalResult performMatmul(OpT op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter, Value &lhs,
                              Value &rhs, Value &output) const{
    auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    auto lhsShape = mx::makeShapeMxCompatible(lhsTy.getShape());
    auto rhsShape = mx::makeShapeMxCompatible(rhsTy.getShape());

    auto lhsElemTy = lhsTy.getElementType();
    auto rhsElemTy = rhsTy.getElementType();

    auto maxInputRank = lhsRank > rhsRank ? lhsRank : rhsRank;

    if (maxInputRank == 1)
      maxInputRank++;

    auto getRankBroadcastedShape = [&](Value tensor,
                                       bool isRHS) -> SmallVector<int64_t> {
      auto tensorTy = llvm::dyn_cast<TensorType>(tensor.getType());
      auto tensorShape = mx::makeShapeMxCompatible(tensorTy.getShape());
      auto tensorRank = tensorTy.getRank();

      SmallVector<int64_t> bcastedShape;

      auto bcastDims = maxInputRank - tensorRank;

      if (isRHS && (tensorRank == 1) && bcastDims) {
        // RHS with rank1 is special. It be synthetically transposed to dim[:-2]
        for (int32_t i = 0; i < bcastDims - 1; i++)
          bcastedShape.push_back(1);
        bcastedShape.push_back(tensorShape[0]);
        bcastedShape.push_back(1);
      } else {
        if (bcastDims > 0) { // rank broadcast
          for (uint32_t i = 0; i < bcastDims; i++)
            bcastedShape.push_back(1);
        }
        for (auto &dim : tensorShape)
          bcastedShape.push_back(dim);
      }
      return bcastedShape;
    };

    auto lhsBroadcastedShape = getRankBroadcastedShape(lhs, false);
    auto lhsBroadcastedTy = mlir::RankedTensorType::get(
        mx::makeShapeLLVMCompatible(lhsBroadcastedShape), lhsElemTy);
    auto rhsBroadcastedShape = getRankBroadcastedShape(rhs, true);
    auto rhsBroadcastedTy = mlir::RankedTensorType::get(
        mx::makeShapeLLVMCompatible(rhsBroadcastedShape), rhsElemTy);

    auto rankBroadcastedLhs =
        lhsRank == maxInputRank
            ? lhs
            : rewriter.create<tosa::ReshapeOp>(
                  op->getLoc(),
                  llvm::dyn_cast<RankedTensorType>(lhsBroadcastedTy),
                  lhs, rewriter.getDenseI64ArrayAttr(lhsBroadcastedShape));

    auto rankBroadcastedRhs =
        rhsRank == maxInputRank
            ? rhs
            : rewriter.create<tosa::ReshapeOp>(
                  op->getLoc(),
                  llvm::dyn_cast<RankedTensorType>(rhsBroadcastedTy),
                  rhs, rewriter.getDenseI64ArrayAttr(rhsBroadcastedShape));

    // TOSA matmul is performed on two 3D inputs and generates a 3D output.
    // Lower ranked tensors are dim-1 reshaped up to 3D
    auto reshapeUpTo3DTensor = [&](Value tensor) -> Value {
      auto tensorTy = llvm::dyn_cast<TensorType>(tensor.getType());
      auto rank = tensorTy.getRank();

      assert(rank <= 3 && "reshapeUpTo3D tensor must receive rank <= 3");
      if (rank == 3)
        return tensor;

      auto shape = mx::makeShapeMxCompatible(tensorTy.getShape());
      SmallVector<int64_t> newShape({1, 1, 1});

      if (rank == 2) { // batchsize = 1
        newShape[1] = shape[0];
        newShape[2] = shape[1];
      } else { // rank 1
        newShape[2] = shape[0];
      }
      auto newType = mlir::RankedTensorType::get(mx::makeShapeLLVMCompatible(newShape),
                                           tensorTy.getElementType());

      return rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          llvm::dyn_cast<RankedTensorType>(newType),
          tensor, rewriter.getDenseI64ArrayAttr(newShape));
    };

    // Check if we need to perform the broadcast on batch dim
    // Not needed if max rank < 3, or if maxrank == 3 and dim[0] matches
    auto needsBatchDimBroadcast = [&]() -> bool {
      if (maxInputRank < 3) {
        return false;
      } else {
        if (maxInputRank == 3 &&
            lhsBroadcastedShape[0] == rhsBroadcastedShape[0]) {
          return false;
        }
        return true;
      }
    };

    auto performBatchDimBroadcast = needsBatchDimBroadcast();

    // Inputs to the tosa.matmul
    Value matmulLhs, matmulRhs;

    using TensorShape_t = struct {
      int64_t dim;
      int64_t shape;
    };

    // Transpose needs to done if transposeDims are not non-monotonically
    // increasing. E.g. [0, 1, 2, 3]: No transpose [1, 0, 2, 3]: Transpose dim0
    // and dim1 The order need not be sequential, since one or more dims may
    // have been removed due to broadcasting.
    auto isTransposeRequired = [](SmallVector<int32_t> transposedDims) -> bool {
      int32_t lastDim = -1;
      for (auto &dim : transposedDims) {
        if (lastDim > dim)
          return true;
        lastDim = dim;
      }
      return false;
    };

    SmallVector<TensorShape_t> batchElems, lhsSqueezedElems, rhsSqueezedElems;

    if (!performBatchDimBroadcast) {
      // Simple with no broadcasting artifacts. Just reshape up to 3D
      matmulLhs = reshapeUpTo3DTensor(rankBroadcastedLhs);
      matmulRhs = reshapeUpTo3DTensor(rankBroadcastedRhs);

    }

    auto matmulLhsShape = mx::makeShapeMxCompatible(
        llvm::dyn_cast<RankedTensorType>(matmulLhs.getType()).getShape());
    auto matmulRhsShape = mx::makeShapeMxCompatible(
        llvm::dyn_cast<RankedTensorType>(matmulRhs.getType()).getShape());

    assert(matmulLhsShape[0] == matmulRhsShape[0] &&
           "tosa.matmul needs same batchsize on LHS and RHS");

    SmallVector<int64_t> matmulOutputShape(
        {matmulLhsShape[0], matmulLhsShape[1], matmulRhsShape[2]});

    Type outputElemTy;
    if (isa<mlir::FloatType>(lhsElemTy)) {
      outputElemTy = lhsElemTy;
    } else { // qint8 emits i32 matmul output
      outputElemTy = rewriter.getIntegerType(32);
    }

    auto mmOutputTy = mlir::RankedTensorType::get(
        mx::makeShapeLLVMCompatible(matmulOutputShape), outputElemTy);

    auto mmOpResult =
        rewriter
            .create<tosa::MatMulOp>(
                op->getLoc(),
                llvm::dyn_cast<RankedTensorType>(mmOutputTy),
                matmulLhs, matmulRhs)
            .getResult();

    // Perform the reshape to output shape. This is always required unless max
    // input rank=3 and there was no broadcasting, in which case the tosa.matmul
    // output itself is correctly shaped.
    bool performOpReshape = !(maxInputRank == 3 && !performBatchDimBroadcast);

    if (performOpReshape) {
      auto computeOpShape = [&](SmallVector<int64_t> &reshapedOpShape,
                                SmallVector<int32_t> &transposedOpDims,
                                SmallVector<int64_t> &transposedOpShapes) {
        if (maxInputRank == 1)
          return;

        if (maxInputRank == 2) {
          if (lhsRank == 2)
            reshapedOpShape.push_back(lhsShape[0]);
          if (rhsRank == 2)
            reshapedOpShape.push_back(rhsShape[1]);
          return;
        }

      };
      SmallVector<int64_t> reshapedOpShape, transposedOpShape;
      SmallVector<int32_t> transposedOpDims;

      computeOpShape(reshapedOpShape, transposedOpDims, transposedOpShape);

      bool opNeedsTranspose = isTransposeRequired(transposedOpDims);

      auto reshapedOpType = mlir::RankedTensorType::get(
          mx::makeShapeLLVMCompatible(reshapedOpShape), outputElemTy);
      auto reshapedOp = rewriter.create<tosa::ReshapeOp>(
          op->getLoc(),
          llvm::dyn_cast<RankedTensorType>(reshapedOpType),
          mmOpResult, rewriter.getDenseI64ArrayAttr(reshapedOpShape));

      if (opNeedsTranspose) {

        std::optional<Value> transposedOpShapeConst =
            tosa::getConstTensor<int32_t>(
                rewriter, op,
                /*vec=*/transposedOpDims,
                /*shape=*/{static_cast<int32_t>(transposedOpDims.size())});

        auto transposedOpType = mlir::RankedTensorType::get(
            mx::makeShapeLLVMCompatible(transposedOpShape), outputElemTy);
        output = rewriter
                     .create<tosa::TransposeOp>(
                         op->getLoc(),
                         llvm::dyn_cast<RankedTensorType>(transposedOpType),
                         reshapedOp.getResult(), transposedOpShapeConst.value())
                     .getResult();

      } else {
        output = reshapedOp.getResult();
      }
    }
    return success();
  }
  virtual LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value lhs, rhs;

    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "Failed to read matmul inputs");

    Value output;

    if (failed(performMatmul(op, adaptor, rewriter, lhs, rhs, output)))
      return rewriter.notifyMatchFailure(op,
                                         "Failed to perform matmul operation");

    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        llvm::dyn_cast<RankedTensorType>(op.getType()),
        output);

    return success();
  }
};

template <typename OpT>
class ConvertMatMulOp : public ConvertMatmulBaseOp<OpT> {
public:
  using ConvertMatmulBaseOp<OpT>::ConvertMatmulBaseOp;
  using OpAdaptor = typename OpT::Adaptor;
  LogicalResult readMatMulInputs(OpT op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {
    lhs = adaptor.getSelf();
    auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());

    rhs = adaptor.getOther();
    auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());

    if (!lhsTy || !rhsTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in TOSA matmul");

    return success();
  }
};

// Implements handling of aten.linear op.
class LinearOpLowering : public ConvertMatmulBaseOp<mx::LinearOp>{
public:
  using ConvertMatmulBaseOp<mx::LinearOp>::ConvertMatmulBaseOp;
  using OpAdaptor = typename mx::LinearOp::Adaptor;
  LogicalResult readMatMulInputs(mx::LinearOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 Value &lhs, Value &rhs) const override {
    lhs = adaptor.getInput();
    auto lhsTy = llvm::dyn_cast<RankedTensorType>(lhs.getType());

    rhs = adaptor.getWeight();
    auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());

    if (!lhsTy || !rhsTy)
      return rewriter.notifyMatchFailure(
          op, "Only ranked tensor types supported in TOSA matmul");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    if (lhsRank != 2 && lhsRank != 3)
      return op.emitError("aten.Linear called but input rank not 2 or 3");
    if (rhsRank != 2 && rhsRank != 3)
      return op.emitError("aten.Linear called but weight rank not 2 or 3");

    // Protection against crash due to unguarded code in TOSA->LinAlg.
    // TODO: This should be handled in TOSA->LinAlg instead.
    if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "aten.Linear needs statically shaped input");

    return success();
  }

  LogicalResult
  matchAndRewrite(mx::LinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs, rhs;

    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "Failed to read matmul inputs");


    auto rhsTy = llvm::dyn_cast<RankedTensorType>(rhs.getType());
    auto rhsRank = rhsTy.getRank();
    auto rhsShape = mx::makeShapeMxCompatible(rhsTy.getShape());
    auto rhsElemTy = rhsTy.getElementType();

    // Create a non-const shape array to transpose dims.
    SmallVector<int64_t> transposedRhsShape;
    for (auto &shape : rhsShape)
      transposedRhsShape.push_back(shape);
    SmallVector<int32_t> transposedRhsDims;
    for (int32_t i = 0; i < rhsRank; i++)
      transposedRhsDims.push_back(i);

    // Swap the last two dims.
    std::swap(transposedRhsShape[rhsRank - 1], transposedRhsShape[rhsRank - 2]);
    std::swap(transposedRhsDims[rhsRank - 1], transposedRhsDims[rhsRank - 2]);

    std::optional<Value> transposedRhsShapeConst =

        tosa::getConstTensor<int32_t>(
            rewriter, op,
            /*vec=*/transposedRhsDims,
            /*shape=*/{static_cast<int32_t>(transposedRhsDims.size())});

    auto transposedRhsType = mlir::RankedTensorType::get(
        mx::makeShapeLLVMCompatible(transposedRhsShape), rhsElemTy);
    rhs = rewriter.create<tosa::TransposeOp>(
        op->getLoc(),
        llvm::dyn_cast<RankedTensorType>(transposedRhsType),
        rhs, transposedRhsShapeConst.value());

    Value matmulOutput;
    if (failed(
            this->performMatmul(op, adaptor, rewriter, lhs, rhs, matmulOutput)))
      return rewriter.notifyMatchFailure(op,
                                         "Failed to perform matmul operation");
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        op,
        llvm::dyn_cast<RankedTensorType>((op.getType())),
        matmulOutput);

    return success();
  }
};
// mx to tosa pass

namespace{
class MxToTosaLowerPass : public mlir::PassWrapper<MxToTosaLowerPass,mlir::OperationPass<mlir::ModuleOp>>{
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MxToTosaLowerPass)
  
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::tosa::TosaDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect, mlir::BuiltinDialect, mlir::arith::ArithDialect>();
  }
  void runOnOperation() final;
};
}

void MxToTosaLowerPass::runOnOperation(){
    mlir::ConversionTarget target(getContext());
    
    target.addIllegalDialect<mx::MxDialect>();
    target.addLegalDialect<mlir::tosa::TosaDialect,mlir::affine::AffineDialect, mlir::BuiltinDialect,
                         mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                         mlir::memref::MemRefDialect, mlir::tensor::TensorDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AddOpLowering, MulOpLowering, AddMulOpLowering, TransposeOpLowering, ReshapeOpLowering, TanhOpLowering, Conv2dOpLowering, MaxPool2dOpLowering, LinearOpLowering>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mx::createLowerToTosaPass() {
  return std::make_unique<MxToTosaLowerPass>();
}
