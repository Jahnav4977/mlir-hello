func.func @test_transpose(%arg0: tensor<2x3xf32>, %arg1 : tensor<2xi32>) -> tensor<3x2xf32> attributes { llvm.emit_c_interface } {
  %0 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "mx.transpose"(%arg0, %0) : (tensor<2x3xf32>, tensor<2xi32>)  -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}