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
    auto new_shape = op.getNewShapeAttr();
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
    patterns.add<AddOpLowering, MulOpLowering, AddMulOpLowering, TransposeOpLowering, ReshapeOpLowering, TanhOpLowering, Conv2dOpLowering>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mx::createLowerToTosaPass() {
  return std::make_unique<MxToTosaLowerPass>();
}
