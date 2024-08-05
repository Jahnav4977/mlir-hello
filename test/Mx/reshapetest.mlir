func.func @test_reshape(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> attributes { llvm.emit_c_interface }{
  %0 = "mx.reshape"(%arg0) { shape = array<i64: 3,2> } : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
