#ifndef MX_DIALECT
#define MX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Mx dialect definition.
//===----------------------------------------------------------------------===//

def Mx_Dialect : Dialect {
    let name = "mx";
    let summary = "A mx out-of-tree MLIR dialect.";
    let description = [{
        This dialect is minimal example to implement mx-world kind of sample code
        for MLIR.
    }];
    let cppNamespace = "::mx";
    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base mx operation definition.
//===----------------------------------------------------------------------===//

class Mx_Op<string mnemonic, list<Trait> traits = []> :
        Op<Mx_Dialect, mnemonic, traits>;

#endif // MX_DIALECT
