func.func @test_transpose_dyn_multiple_2d(%arg0: tensor<2x3xf32>, %arg1 : tensor<2xi32>) -> tensor<3x2xf32> attributes { llvm.emit_c_interface } {
  %0 = "mx.transpose"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2xi32>)  -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}