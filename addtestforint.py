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
        input = """func.func @test_add(%arg0: i32, %arg1: i32) -> i32 attributes { llvm.emit_c_interface } {
  %0 = "mx.addi"(%arg0, %arg1) : (i32, i32) -> i32
  return %0 : i32
}"""
        mx_process = subprocess.run(
            ['build/bin/mx-opt'],
            input=input,
            capture_output=True,
            encoding='ascii',
        )
        mlir_LLVM_txt = mx_process.stderr
        print(mlir_LLVM_txt)
        module = Module.parse(mlir_LLVM_txt)

        '''arg1 = np.array(10).astype(np.int32)
        arg2 = np.array(20).astype(np.int32)
        arg1_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_scalar_memref_descriptor(arg1))
        )
        arg2_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_scalar_memref_descriptor(arg2))
        )
        res = np.array(0).astype(np.int32)
        res_memref_ptr = ctypes.pointer(ctypes.pointer(
            make_nd_memref_descriptor(0, np.ctypeslib.as_ctypes_type(np.int32))()
        ))

        execution_engine = ExecutionEngine(
            module,
            opt_level=3
        )
        execution_engine.invoke("test_add", res_memref_ptr, arg1_memref_ptr, arg2_memref_ptr)
        res = res_memref_ptr.contents.contents.value

        res_tensor = torch.tensor(res)
        arg1_tensor = torch.tensor(arg1)
        arg2_tensor = torch.tensor(arg2)
        ans_tensor = arg1_tensor + arg2_tensor

        print(f"Result: {res}")
        print(f"Expected: {ans_tensor.item()}")
        print(f"Actual: {res_tensor.item()}")'''

execute()
