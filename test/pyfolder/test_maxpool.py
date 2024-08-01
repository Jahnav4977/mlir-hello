import sys
import torch
import subprocess
import pytest
import numpy as np
from mlir.execution_engine import ExecutionEngine, ctypes 
from mlir.ir import Context, Module
from mlir.passmanager import PassManager
from mlir.runtime import *


def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()

def execute_maxpool2d():
    with Context():
        input = """func.func @test_maxpool2d(%arg0: tensor<3x3x10x10xf32>) -> tensor<3x3x9x9xf32> attributes { llvm.emit_c_interface } {
  %0 = "mx.maxpool2d"(%arg0) {kernel = array<i64: 2,2>, stride = array<i64: 1,1>, pad = array<i64: 0,0,0,0>} : (tensor<3x3x10x10xf32>) -> tensor<3x3x9x9xf32>
  return %0 : tensor<3x3x9x9xf32>
}"""
        mx_process = subprocess.run(
            ['../../build/bin/mx-opt'],
            input=input,
            capture_output=True,
            encoding='ascii',
        )
        
        mlir_LLVM_txt = mx_process.stderr
        module=Module.parse(mlir_LLVM_txt)
        print(module)
        
        arg0 = np.random.rand(3, 3, 10, 10).astype(np.float32)
        arg0_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg0)))
        output_shape = (3, 3, 9, 9)
        res = np.zeros(output_shape, dtype=np.float32)
        res_memref_ptr = ctypes.pointer(ctypes.pointer(make_nd_memref_descriptor(len(output_shape), np.ctypeslib.as_ctypes_type(np.float32))()))
        
        execution_engine = ExecutionEngine(module)
        execution_engine.invoke("test_maxpool2d", res_memref_ptr, arg0_memref_ptr)
        
        res = ranked_memref_to_numpy(res_memref_ptr[0])
        res_tensor = torch.from_numpy(res)
        arg0_tensor = torch.from_numpy(arg0)
        
        maxpool2d = torch.nn.functional.max_pool2d(arg0_tensor, kernel_size=2, stride=1, padding=0)
        assert torch.allclose(res_tensor, maxpool2d, atol=1e-5), f"Expected {maxpool2d} but got {res_tensor}"
        
def test_maxpool2d():
    execute_maxpool2d()
    

if __name__ == "__main__":
    test_maxpool2d()
