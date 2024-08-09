//./build/bin/mx-opt --mlir-print-ir-after-all ./test/Mx/multest.mlir

func.func @test_mul(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> attributes { llvm.emit_c_interface } {
  %0 = "mx.mul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}