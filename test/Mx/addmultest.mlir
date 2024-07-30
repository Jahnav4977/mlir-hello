

func.func @test_add(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>, %arg2: tensor<2x3xf64>) attributes { llvm.emit_c_interface } {
  %0 = "mx.addmul"(%arg0, %arg1, %arg2) : (tensor<2x3xf64>, tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  return 
}