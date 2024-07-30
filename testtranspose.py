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
        input = """func.func @test_add(%arg0: tensor<2x3xf32>, %arg1: tensor<2xi32>) -> tensor<3x2xf32> attributes { llvm.emit_c_interface } {
  %0 = "mx.transpose"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}"""
        mx_process = subprocess.run(
            ['build/bin/mx-opt'],
            input=input,
            capture_output=True,
            encoding='ascii',
        )
        
        mlir_LLVM_txt = mx_process.stderr
        print(mlir_LLVM_txt)
        module=Module.parse(mlir_LLVM_txt)
        
        
execute()
