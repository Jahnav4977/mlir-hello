import sys
import subprocess
import numpy as np
import torch
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
  %0 = "mx.addmul"(%arg0, %arg1, %arg2) : (tensor<2x3xf64>, tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}"""
        hello_process = subprocess.run(
            ['../../build/bin/mx-opt'],
            input=input,
            capture_output=True,
            encoding='ascii',
        )
        mlir_LLVM_txt = hello_process.stderr
        module = Module.parse(mlir_LLVM_txt)
        
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
            ctypes.pointer(get_ranked_memref_descriptor(arg3))
        )
        addmulres = np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float64)
        addmulres_memref_ptr = ctypes.pointer(ctypes.pointer(
            make_nd_memref_descriptor(2, np.ctypeslib.as_ctypes_type(np.float64))()
        ))
        execution_engine = ExecutionEngine(module)
        execution_engine.invoke("test_addmul", addmulres_memref_ptr, arg1_memref_ptr, arg2_memref_ptr, arg3_memref_ptr)
        addmulres = ranked_memref_to_numpy(addmulres_memref_ptr[0])
        print(addmulres)
        addmulres_tensor = torch.from_numpy(addmulres)
        print(addmulres_tensor)
        arg1_tensor = torch.from_numpy(arg1)
        arg2_tensor = torch.from_numpy(arg2)
        arg3_tensor = torch.from_numpy(arg3)
        ans_tensor = (arg1_tensor + arg2_tensor) * arg3_tensor
        print(ans_tensor)
        assert torch.allclose(addmulres_tensor, ans_tensor), f"Expected {ans_tensor} but got {addmulres_tensor}"

def test_answer():
    execute()

if __name__ == "__main__":
    test_answer()
