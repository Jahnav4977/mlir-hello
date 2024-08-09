//./build/bin/mx-opt --mlir-print-ir-after-all ./test/Mx/reshapetest2.mlir

func.func @test_reshape(%arg0: tensor<2x3xf32>) -> tensor<1x2x3xf32> attributes { llvm.emit_c_interface }{
  %0 = "mx.reshape"(%arg0) { shape = array<i64: 1,2,3> } : (tensor<2x3xf32>) -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>
}
