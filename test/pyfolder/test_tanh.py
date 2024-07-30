import sys
import torch
import subprocess
import numpy as np
from mlir.execution_engine import ExecutionEngine, ctypes 
from mlir.ir import Context, Module
from mlir.passmanager import PassManager
from mlir.runtime import *


def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()

def execute():
    with Context():
        input = """func.func @test_tanh(%arg0: tensor<2x3xf64>) -> tensor<2x3xf64> attributes { llvm.emit_c_interface } {
  %0 = "mx.tanh"(%arg0) : (tensor<2x3xf64>) -> tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}"""
        hello_process = subprocess.run(
            ['../../build/bin/mx-opt'],
            input=input,
            capture_output=True,
            encoding='ascii',
        )
        
        mlir_LLVM_txt = hello_process.stderr
        module=Module.parse(mlir_LLVM_txt)
        arg1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float64)
        arg1_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(arg1))
        )
        res = np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float64)
        res_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(res))
        )
        execution_engine=ExecutionEngine(module)
        execution_engine.invoke("test_tanh", res_memref_ptr, arg1_memref_ptr)
        res = ranked_memref_to_numpy(res_memref_ptr[0])
        print(res)
        res_tensor=torch.from_numpy(res)
        print(res_tensor)
        arg1_tensor=torch.from_numpy(arg1)
        ans_tensor=torch.tanh(arg1_tensor)
        print(ans_tensor)
        assert torch.allclose(res_tensor, ans_tensor), f"Expected {ans_tensor} but got {res_tensor}"

def test_answer():
    execute()
    

if __name__ == "__main__":
    test_answer()