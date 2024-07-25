func.func @main(%arg0: memref<2x3xf64>) attributes { llvm.emit_c_interface }{
    %0 = "hello.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %1 = "hello.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %2 = "hello.add"(%0, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    return 
}