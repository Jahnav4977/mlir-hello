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

def execute_reshape2():
    with Context():
        with open('../Mx/reshapetest2.mlir', 'r') as f:
            input = f.read()
        hello_process = subprocess.run(
            ['../../build/bin/mx-opt'],
            input=input,
            capture_output=True,
            encoding='ascii',
        )
        
        mlir_LLVM_txt = hello_process.stderr
        module=Module.parse(mlir_LLVM_txt)
        print(module)
        
        arg1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float32)
        arg1_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(arg1))
        )
        res = np.array([[[0, 0, 0], [0, 0, 0]]]).astype(np.float32)
        res_memref_ptr = ctypes.pointer(ctypes.pointer(
            make_nd_memref_descriptor(3, np.ctypeslib.as_ctypes_type(np.float32))()
        ))
        shared_libs = [
                "/home/jahnav/Desktop/llvm-project/build/lib/libmlir_c_runner_utils.so",
                "/home/jahnav/Desktop/llvm-project/build/lib/libmlir_runner_utils.so",
            ]
        execution_engine=ExecutionEngine(module, opt_level=3, shared_libs=shared_libs)
        execution_engine.invoke("test_reshape", res_memref_ptr, arg1_memref_ptr)
        res = ranked_memref_to_numpy(res_memref_ptr[0])
        res_tensor=torch.from_numpy(res)
        print(res)
        arg1_tensor = torch.from_numpy(arg1)
        reshaped = arg1_tensor.reshape(1,2,3)
        assert torch.allclose(res_tensor, reshaped, atol=1e-5), f"Expected {reshaped} but got {res_tensor}" 

def test_reshape2():
    execute_reshape2()
    

if __name__ == "__main__":
    test_reshape2()