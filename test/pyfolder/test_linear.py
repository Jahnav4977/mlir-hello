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

def execute_linear():
    with Context():
        with open('../Mx/lineartest.mlir', 'r') as f:
            input = f.read()
        hello_process = subprocess.run(
            ['../../build/bin/mx-opt'],
            input=input,
            capture_output=True,
            encoding='ascii',
        )
        
        mlir_LLVM_txt = hello_process.stderr
        module=Module.parse(mlir_LLVM_txt)

        arg0 = np.random.rand(5, 2).astype(np.float32)
        arg1 = np.random.rand(3, 2).astype(np.float32)
        
        arg0_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg0)))
        arg1_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg1)))
        
        output_shape = (5, 3)
        res = np.zeros(output_shape, dtype=np.float32)
        res_memref_ptr = ctypes.pointer(ctypes.pointer(make_nd_memref_descriptor(len(output_shape), np.ctypeslib.as_ctypes_type(np.float32))()))
        execution_engine=ExecutionEngine(module)
        execution_engine.invoke("test_linear", res_memref_ptr, arg0_memref_ptr, arg1_memref_ptr)
        res = ranked_memref_to_numpy(res_memref_ptr[0])
        print(res)
        res_tensor=torch.from_numpy(res)
        print(res_tensor)
        arg0_tensor = torch.from_numpy(arg0)
        arg1_tensor = torch.from_numpy(arg1)
        m=torch.nn.Linear(2,3,bias=False)
        m.weight.data = arg1_tensor
        ans_tensor=m(arg0_tensor)
        print(ans_tensor)
        assert torch.allclose(res_tensor, ans_tensor), f"Expected {ans_tensor} but got {res_tensor}" 

def test_linear():
    execute_linear()
    

if __name__ == "__main__":
    test_linear()

