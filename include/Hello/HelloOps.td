#ifndef HELLO_OPS
#define HELLO_OPS

include "HelloDialect.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : Hello_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "hello.constant"()
      { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
      : () -> tensor<2x3xf64>
    ```
  }];

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

//  let parser = [{ return ::parseConstantOp(parser, result); }];

  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
}

def PrintOp : Hello_Op<"print", [Pure]> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def WorldOp : Hello_Op<"world", [Pure]> {
    let summary = "print Hello, World";
    let description = [{
        The "world" operation prints "Hello, World", and produces
        no results.
    }];
}

def AddOp : Hello_Op<"add", [Pure]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor:$output);
  
  //let hasCustomAssemblyFormat = 1;
  
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs,"mlir::Value":$rhs)>
  ];
}

def MulOp : Hello_Op<"mul", [Pure]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor:$output);
  
  //let hasCustomAssemblyFormat = 1;
  
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs,"mlir::Value":$rhs)>
  ];
}

def AddMulOp : Hello_Op<"addmul", [Pure]>{
  let summary = "element-wise addition and then multiplication";
  let description = [{
    The "addmul" operation performs element-wise addition between two tensors and multiplies the resultant output  with itself
  }];

  let arguments = (ins F64Tensor:$first, F64Tensor:$second, F64Tensor:$third);
  let results = (outs F64Tensor:$output);

  let builders = [
    OpBuilder<(ins "mlir::Value":$first,"mlir::Value":$second,"mlir::Value":$third)>
  ];
}

#endif // HELLO_OPS