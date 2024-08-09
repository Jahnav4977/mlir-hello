// ./build/bin/mx-opt --mlir-print-ir-after-all ./test/Mx/addtest.mlir

func.func @test_add(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) -> tensor<2x3xf64> attributes { llvm.emit_c_interface } {
  %0 = "mx.add"(%arg0, %arg1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}