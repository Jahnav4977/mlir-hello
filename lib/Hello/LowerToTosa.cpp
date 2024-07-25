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
#include "Hello/HelloDialect.h"
#include "Hello/HelloOps.h"
#include "Hello/HelloPasses.h"

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
    
class AddOpLowering : public OpRewritePattern<hello::AddOp>{
    using OpRewritePattern<hello::AddOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(hello::AddOp op,PatternRewriter &rewriter) const final{
        auto lhs=op.getLhs();
        auto rhs=op.getRhs();
        auto output=op.getResult();
        auto outputType=llvm::dyn_cast<RankedTensorType>(output.getType());
        rewriter.replaceOpWithNewOp<mlir::tosa::AddOp>(op,outputType,lhs,rhs);
    return success();
    }
};

class MulOpLowering : public OpRewritePattern<hello::MulOp>{
    using OpRewritePattern<hello::MulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(hello::MulOp op,PatternRewriter &rewriter) const final{
        auto lhs=op.getLhs();
        auto rhs=op.getRhs();
        auto output=op.getResult();
        auto outputType=llvm::dyn_cast<RankedTensorType>(output.getType());
        rewriter.replaceOpWithNewOp<mlir::tosa::MulOp>(op,outputType,lhs,rhs,0);
    return success();
    }
};

class AddMulOpLowering : public OpRewritePattern<hello::AddMulOp>{
   using OpRewritePattern<hello::AddMulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(hello::AddMulOp op,PatternRewriter &rewriter) const final{
       auto first=op.getFirst();
       auto second=op.getSecond();
       auto third=op.getThird();
       auto output=op.getResult();
       auto outputType=llvm::dyn_cast<RankedTensorType>(output.getType());
       Value temp;
       temp = rewriter.create<hello::AddOp>(op->getLoc(), outputType, first,second).getResult();
       rewriter.replaceOpWithNewOp<hello::MulOp>(op,outputType,temp,third);
    return success();
    }
 };
}

// hello to tosa pass

namespace{
class HelloToTosaLowerPass : public mlir::PassWrapper<HelloToTosaLowerPass,mlir::OperationPass<mlir::ModuleOp>>{
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HelloToTosaLowerPass)
  
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::tosa::TosaDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect, mlir::BuiltinDialect, mlir::arith::ArithDialect>();
  }
  void runOnOperation() final;
};
}

void HelloToTosaLowerPass::runOnOperation(){
    mlir::ConversionTarget target(getContext());
    
    target.addIllegalDialect<hello::HelloDialect>();
    target.addLegalDialect<mlir::tosa::TosaDialect,mlir::affine::AffineDialect, mlir::BuiltinDialect,
                         mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                         mlir::memref::MemRefDialect, mlir::tensor::TensorDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AddOpLowering, MulOpLowering, AddMulOpLowering>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> hello::createLowerToTosaPass() {
  return std::make_unique<HelloToTosaLowerPass>();
}
