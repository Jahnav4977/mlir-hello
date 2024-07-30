#ifndef MX_TYPES
#define MX_TYPES

include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// Mx Type Definitions.
//===----------------------------------------------------------------------===//

def Mx_Int32 : I<32>;
def Mx_Int64 : I<64>;

def Mx_Int : AnyTypeOf<[AnyUnsignedInteger,
                          AnySignlessInteger]>;

def Mx_AnyNumber : AnyTypeOf<[Mx_Int, AnyFloat],
                               "number">;
def Mx_Tensor : TensorOf<[Mx_AnyNumber]>;

def Mx_Int32Or64 : AnyTypeOf<[Mx_Int32,
                   	        Mx_Int64]>;

def Mx_Int32Or64Tensor : TensorOf<[Mx_Int32Or64]>;
def Mx_FloatTensor : TensorOf<[AnyFloat]>;

// custom dimensions

def Mx_UnrankedTensor : UnrankedTensorOf<[Mx_AnyNumber]>;

def Mx_Tensor1Dto6D : AnyTypeOf<[
  Mx_UnrankedTensor, TensorRankOf<[Mx_AnyNumber], [1,2,3,4,5,6]>]>;

#endif //MX_TYPES