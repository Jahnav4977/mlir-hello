#ifndef MX_TYPES
#define MX_TYPES

include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// Mx Type Definitions.
//===----------------------------------------------------------------------===//
def Mx_Int4 : I<4>;
def Mx_Int8 : I<8>;
def Mx_Int16 : I<16>;
def Mx_Int32 : I<32>;
def Mx_Int64 : I<64>;

def Mx_Int : AnyTypeOf<[AnyUnsignedInteger,
                          AnySignlessInteger]>;

def Mx_AnyNumber : AnyTypeOf<[Mx_Int, AnyFloat],
                               "number">;

// For weight tensors from tosa::Conv2DOp, tosa::Conv3DOp,
// tosa::DepthwiseConv2DOp, tosa::TransposeConv2DOp, tosa::FullyConnectedOp
def Mx_Weight : AnyTypeOf<[Mx_Int4, Mx_Int8, AnyFloat]>;

def Mx_Tensor : TensorOf<[Mx_AnyNumber]>;

def Mx_Int32Or64 : AnyTypeOf<[Mx_Int32,
                   	        Mx_Int64]>;

def Mx_Int32Or64Tensor : TensorOf<[Mx_Int32Or64]>;
def Mx_FloatTensor : TensorOf<[AnyFloat]>;

// custom dimensions

def Mx_UnrankedTensor : UnrankedTensorOf<[Mx_AnyNumber]>;

def Mx_Tensor1D : AnyTypeOf<[Mx_UnrankedTensor, 1DTensorOf<[Mx_AnyNumber]>], "1-d tensor", "::mlir::TensorType">;
def Mx_Tensor4D : AnyTypeOf<[Mx_UnrankedTensor, 4DTensorOf<[Mx_AnyNumber]>], "4-d tensor", "::mlir::TensorType">;

def Mx_Tensor1Dto6D : AnyTypeOf<[
  Mx_UnrankedTensor, TensorRankOf<[Mx_AnyNumber], [1,2,3,4,5,6]>]>;

// Attributes 
def Mx_IntArrayAttr2 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>]>;
def Mx_IntArrayAttr4 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>;


#endif //MX_TYPES