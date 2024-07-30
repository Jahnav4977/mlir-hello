func.func @reshape_test(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %reshape = "mx.reshape"(%arg0,{ new_shape = array<i64: 3,2> } : (tensor<2x3xf32>)) -> tensor<3x2xf32>
  return %reshape : tensor<3x2xf32>
}
