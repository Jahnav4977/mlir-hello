func.func @test_forward(%arg0: tensor<1x1x28x28xf32>, %arg1: tensor<20x1x5x5xf32>, %arg2: tensor<20xf32>, %arg3:tensor<50x20x5x5xf32>, %arg4:tensor<50xf32>, %arg5:tensor<500x800xf32>, %arg6:tensor<10x500xf32>) -> tensor<1x10xf32> attributes { llvm.emit_c_interface } {
  %0 = "mx.conv2d"(%arg0, %arg1, %arg2) { dilation = array<i64: 1,1>, pad = array<i64: 0,0,0,0>, stride = array<i64: 1,1>} : (tensor<1x1x28x28xf32>, tensor<20x1x5x5xf32>, tensor<20xf32>) -> tensor<1x20x24x24xf32>
  %1 = "mx.tanh"(%0) : (tensor<1x20x24x24xf32>) -> (tensor<1x20x24x24xf32>)
  %2 = "mx.maxpool2d"(%1) {kernel = array<i64: 2,2>, stride = array<i64: 2,2>, pad = array<i64: 0,0,0,0>} : (tensor<1x20x24x24xf32>) -> tensor<1x20x12x12xf32>
  %3 = "mx.conv2d"(%2, %arg3, %arg4) { dilation = array<i64: 1,1>, pad = array<i64: 0,0,0,0>, stride = array<i64: 1,1>} : (tensor<1x20x12x12xf32>, tensor<50x20x5x5xf32>, tensor<50xf32>) -> tensor<1x50x8x8xf32>
  %4 = "mx.tanh"(%3) : (tensor<1x50x8x8xf32>) -> (tensor<1x50x8x8xf32>)
  %5 = "mx.maxpool2d"(%4) {kernel = array<i64: 2,2>, stride = array<i64: 2,2>, pad = array<i64: 0,0,0,0>} : (tensor<1x50x8x8xf32>) -> tensor<1x50x4x4xf32>
  %6 = "mx.reshape"(%5) { shape = array<i64: 1,800> } : (tensor<1x50x4x4xf32>) -> tensor<1x800xf32>
  %7 = "mx.linear"(%6, %arg5) : (tensor<1x800xf32>, tensor<500x800xf32>) -> tensor<1x500xf32>
  %8 = "mx.tanh"(%7) : (tensor<1x500xf32>) -> (tensor<1x500xf32>)
  %9 = "mx.linear"(%8, %arg6) : (tensor<1x500xf32>, tensor<10x500xf32>) -> tensor<1x10xf32>
  %10 = "mx.tanh"(%9) : (tensor<1x10xf32>) -> (tensor<1x10xf32>)
  return %10 : tensor<1x10xf32>
}

