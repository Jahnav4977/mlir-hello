import sys
import numpy as np
from mlir.execution_engine import ExecutionEngine, ctypes 
from mlir.ir import Context, Module
from mlir.passmanager import PassManager
from mlir.runtime import get_ranked_memref_descriptor


def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()

def lower_tosa_to_llvm(module):
    pipeline="""
    builtin.module(
        func.func(tosa-to-linalg),
        one-shot-bufferize{allow-unknown-ops},
        func-bufferize,convert-vector-to-scf,
        func.func(convert-linalg-to-loops,lower-affine),
        convert-scf-to-cf,canonicalize,
        convert-arith-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,
        finalize-memref-to-llvm,
        convert-index-to-llvm,reconcile-unrealized-casts
    )
    """
    pm = PassManager.parse(pipeline)
    pm.run(module)
    return module

def execute():
    with Context():
        input="""
        func.func @test_add(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> attributes { llvm.emit_c_interface } {
            %0 = "tosa.add"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
            return %0 : tensor<2xf32>
        }
        """
        tosa_module = Module.parse(input)
        arg1 = np.array([1.0,2.0]).astype(np.float32)
        arg2 = np.array([2.0,3.0]).astype(np.float32)
        res = np.array([0.0,0.0]).astype(np.float32)

        arg1_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg1)))
        arg2_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg2)))
        res_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(res)))

        execution_engine=ExecutionEngine(
            lower_tosa_to_llvm(tosa_module),
            opt_level=3
        )
        
        execution_engine.invoke(
            "test_add", arg1_memref_ptr, arg2_memref_ptr, res_memref_ptr
        )
        # expected output 
        # CHECK: [1.0, 2.0] + [2.0, 3.0] = [3.0, 5.0]
        log("{0} + {1} = {2}".format(arg1, arg2, res))
        # actual output ( getting same initial value)
        # [1.0, 2.0] + [2.0, 3.0] = [0.0, 0.0]  
        
execute()