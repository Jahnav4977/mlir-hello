import sys
import subprocess
import numpy as np
from mlir.execution_engine import ExecutionEngine, ctypes 
from mlir.ir import Context, Module
from mlir.passmanager import PassManager
from mlir.runtime import *


def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()

def lowerToLLVM(module):
    pm = PassManager.parse(
        "builtin.module(lower-affine,convert-scf-to-cf,convert-func-to-llvm,convert-cf-to-llvm,finalize-memref-to-llvm,reconcile-unrealized-casts)"
    )
    pm.run(module.operation)
    return module

def execute():
    with Context():
        input = """func.func @test_addmul(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>, %arg2: tensor<2x3xf64>) -> tensor<2x3xf64> attributes { llvm.emit_c_interface } {
  %0 = "hello.addmul"(%arg0, %arg1, %arg2) : (tensor<2x3xf64>, tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}"""
        hello_process = subprocess.run(
            ['build/bin/hello-opt'],
            input=input,
            capture_output=True,
            encoding='ascii',
        )
        mlir_LLVM_txt = hello_process.stderr
        module=Module.parse(mlir_LLVM_txt)
        
        arg1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float64)
        arg2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float64)
        arg3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float64)
        arg1_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(arg1))
        )
        arg2_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(arg2))
        )
        arg3_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(arg2))
        )
        res = np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float64)
        res_memref_ptr = ctypes.pointer(ctypes.pointer(
            make_nd_memref_descriptor(2, np.ctypeslib.as_ctypes_type(np.float64))()
        ))
        execution_engine=ExecutionEngine(module)
        execution_engine.invoke("test_addmul", res_memref_ptr, arg1_memref_ptr, arg2_memref_ptr, arg3_memref_ptr)
        res = ranked_memref_to_numpy(res_memref_ptr[0])
        # expected output 
        # CHECK: [1.0, 2.0] + [2.0, 3.0] = [3.0, 5.0]
        log("{0} + {1} * {2} = {3}".format(arg1, arg2, arg3, res))
        #print(mlir_LLVM_txt)
        
execute()
