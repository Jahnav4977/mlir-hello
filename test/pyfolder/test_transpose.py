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

def execute_transpose():
    with Context():
        with open('../Mx/transposetest.mlir', 'r') as f:
            input = f.read()
        mx_process = subprocess.run(
            ['../../build/bin/mx-opt'],
            input=input,
            capture_output=True,
            encoding='ascii',
        )
        
        mlir_LLVM_txt = mx_process.stderr
        module=Module.parse(mlir_LLVM_txt)
        
        arg1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(np.float32)
        arg1_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(arg1))
        )
        res = np.array([[0, 0], [0,0], [0, 0]]).astype(np.float32)
        res_memref_ptr = ctypes.pointer(ctypes.pointer(
            make_nd_memref_descriptor(2, np.ctypeslib.as_ctypes_type(np.float32))()
        ))
        execution_engine=ExecutionEngine(module)
        execution_engine.invoke("test_transpose", res_memref_ptr, arg1_memref_ptr)
        res = ranked_memref_to_numpy(res_memref_ptr[0])
        res_tensor = torch.from_numpy(res)
        arg1_tensor = torch.from_numpy(arg1)
        
        transpose = torch.transpose(arg1_tensor, 1, 0)
        assert torch.allclose(res_tensor, transpose, atol=1e-5), f"Expected {transpose} but got {res_tensor}"

def test_transpose():
    execute_transpose()
    

if __name__ == "__main__":
    test_transpose()

