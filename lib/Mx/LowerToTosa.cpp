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
    patterns.add<AddOpLowering, MulOpLowering, AddMulOpLowering, TransposeOpLowering, ReshapeOpLowering, TanhOpLowering>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mx::createLowerToTosaPass() {
  return std::make_unique<MxToTosaLowerPass>();
}
