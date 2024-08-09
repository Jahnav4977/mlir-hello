//./build/bin/mx-opt --mlir-print-ir-after-all ./test/Mx/maxpool2dtest.mlir

func.func @test_maxpool2d(%arg0: tensor<3x3x10x10xf32>) -> tensor<3x3x9x9xf32> attributes { llvm.emit_c_interface } {
  %0 = "mx.maxpool2d"(%arg0) {kernel = array<i64: 2,2>, stride = array<i64: 1,1>, pad = array<i64: 0,0,0,0>} : (tensor<3x3x10x10xf32>) -> tensor<3x3x9x9xf32>
  return %0 : tensor<3x3x9x9xf32>
}