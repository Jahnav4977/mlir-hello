#ifndef MX_OPS
#define MX_OPS

include "MxDialect.td"
include "MxType.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : Mx_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "mx.constant"()
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

def PrintOp : Mx_Op<"print", [Pure]> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def WorldOp : Mx_Op<"world", [Pure]> {
    let summary = "print Mx, World";
    let description = [{
        The "world" operation prints "Mx, World", and produces
        no results.
    }];
}

def AddOp : Mx_Op<"add", [Pure]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  
  let arguments = (ins
    Mx_Tensor:$lhs,
    Mx_Tensor:$rhs
  );
  let results = (outs Mx_Tensor:$output);
  
  //let hasCustomAssemblyFormat = 1;
  
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs,"mlir::Value":$rhs)>
  ];
}

def MulOp : Mx_Op<"mul", [Pure]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins
    Mx_FloatTensor:$lhs,
    Mx_FloatTensor:$rhs
  );
  let results = (outs Mx_FloatTensor:$output);
  
  //let hasCustomAssemblyFormat = 1;
  
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs,"mlir::Value":$rhs)>
  ];
}

def AddMulOp : Mx_Op<"addmul", [Pure]>{
  let summary = "element-wise addition and then multiplication";
  let description = [{
    The "addmul" operation performs element-wise addition between two tensors and multiplies the resultant output  with itself
  }];

  let arguments = (ins
    Mx_FloatTensor:$first,
    Mx_FloatTensor:$second,
    Mx_FloatTensor:$third
  );
  let results = (outs Mx_FloatTensor:$output);

  let builders = [
    OpBuilder<(ins "mlir::Value":$first,"mlir::Value":$second,"mlir::Value":$third)>
  ];
}

def ReshapeOp: Mx_Op<"reshape", [Pure]> {
  let summary = "Reshape operator";

  let description = [{
    Returns a tensor with the same type/values as the input, with a new shape
    specified by the shape argument. Reshape may operate on tensors of any rank.
    No data conversion happens during a reshape operation.
  }];

  let arguments = (ins
    Mx_Tensor:$input1,
    DenseI64ArrayAttr:$new_shape
  );

  let results = (outs
    Mx_Tensor:$output
  );
}

def TransposeOp : Mx_Op<"transpose",[Pure]>{
  let summary = "Transpose operator";

  let description = [{
    Permutes the dimensions based on perm.
  }];

  let arguments = (ins
    Mx_Tensor1Dto6D:$input1,
    Mx_Int32Or64Tensor:$perms
  );

  let results = (
    outs Mx_Tensor1Dto6D:$output
  );
}

def TanhOp : Mx_Op<"tanh",[Pure]>{
  let summary = "Tanh operator";

  let description = [{
    performs hyperbolic activation function.
  }];

  let arguments = (ins
    Mx_Tensor:$input
  );
  let results = (outs Mx_Tensor:$output);
}

def Conv2dOp : Mx_Op<"conv2d", [Pure]>{
  let summary = "Conv2d Operator";

  let description = [{
    performs convolution on 2d tensor.
  }];

  let arguments = (ins 
    Mx_Tensor4D:$input,
    4DTensorOf<[Mx_Weight]>:$weight,
    Mx_Tensor1D:$bias,

    Mx_IntArrayAttr4:$pad,
    Mx_IntArrayAttr2:$stride,
    Mx_IntArrayAttr2:$dilation
  );
  let results = (outs 
    Mx_Tensor4D:$output
  );
}

#endif // MX_OPS