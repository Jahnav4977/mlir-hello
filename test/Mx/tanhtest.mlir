//./build/bin/mx-opt --mlir-print-ir-after-all ./test/Mx/tanhtest.mlir

func.func @test_tanh(%arg0: tensor<2x3xf64>) -> tensor<2x3xf64> attributes { llvm.emit_c_interface } {
  %0 = "mx.tanh"(%arg0) : (tensor<2x3xf64>) -> tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}