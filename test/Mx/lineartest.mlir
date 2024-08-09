// ./build/bin/mx-opt --mlir-print-ir-after-all ./test/Mx/lineartest.mlir

func.func @test_linear(%arg0: tensor<5x2xf32>, %arg1: tensor<3x2xf32>) -> tensor<5x3xf32> attributes { llvm.emit_c_interface } {
  %0 = "mx.linear"(%arg0, %arg1) : (tensor<5x2xf32>, tensor<3x2xf32>) -> tensor<5x3xf32>
  return %0 : tensor<5x3xf32>
}