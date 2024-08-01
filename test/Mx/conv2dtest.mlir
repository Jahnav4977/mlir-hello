func.func @test_conv2d(%arg0: tensor<3x3x10x10xf32>, %arg1: tensor<3x3x2x2xf32>, %arg2: tensor<3xf32>) -> tensor<3x3x9x9xf32> attributes { llvm.emit_c_interface } {
  %0 = "mx.conv2d"(%arg0, %arg1, %arg2) { dilation = array<i64: 1,1>, pad = array<i64: 0,0,0,0>, stride = array<i64: 1,1>} : (tensor<3x3x10x10xf32>, tensor<3x3x2x2xf32>, tensor<3xf32>) -> tensor<3x3x9x9xf32>
  return %0 : tensor<3x3x9x9xf32>
}