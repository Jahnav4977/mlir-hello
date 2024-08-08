import sys
import torch
import subprocess
import numpy as np
from mlir.execution_engine import ExecutionEngine, ctypes 
from mlir.ir import Context, Module
from mlir.passmanager import PassManager
from mlir.runtime import *
from torchvision import datasets, transforms


def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()

def execute_forward():
    with Context():
        with open('../Mx/forwardtest.mlir', 'r') as f:
            input = f.read()
        mx_process = subprocess.run(
            ['../../build/bin/mx-opt'],
            input=input,
            capture_output=True,
            encoding='ascii',
        )
        
        mlir_LLVM_txt = mx_process.stderr
        module=Module.parse(mlir_LLVM_txt)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)
        mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=True)
        
        # Get a single image and its label from the dataset
        mnist_image, mnist_label = next(iter(mnist_loader))
        print(mnist_image)
        hey = mnist_label.numpy().astype(np.float32)
        print(hey)
        # Convert the image to a NumPy array
        arg0 = mnist_image.numpy().astype(np.float32)
        print(arg0)
        #arg0 = np.random.rand(1, 1, 28, 28).astype(np.float32)
        arg1 = np.random.rand(20, 1, 5, 5).astype(np.float32)
        arg2 = np.random.rand(20).astype(np.float32)
        arg3 = np.random.rand(50, 20, 5, 5).astype(np.float32)
        arg4 = np.random.rand(50).astype(np.float32)
        arg5 = np.random.rand(500, 800).astype(np.float32)
        arg6 = np.random.rand(10, 500).astype(np.float32)
        
        arg0_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg0)))
        arg1_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg1)))
        arg2_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg2)))
        arg3_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg3)))
        arg4_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg4)))
        arg5_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg5)))
        arg6_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg6)))
        
        output_shape = (1, 50, 4, 4)
        res = np.zeros(output_shape, dtype=np.float32)
        res_memref_ptr = ctypes.pointer(ctypes.pointer(make_nd_memref_descriptor(len(output_shape), np.ctypeslib.as_ctypes_type(np.float32))()))

        execution_engine = ExecutionEngine(module)
        execution_engine.invoke("test_forward", res_memref_ptr, arg0_memref_ptr, arg1_memref_ptr, arg2_memref_ptr, arg3_memref_ptr, arg4_memref_ptr, arg5_memref_ptr, arg6_memref_ptr)

        res = ranked_memref_to_numpy(res_memref_ptr[0])
        res_tensor = torch.from_numpy(res)
        print(res_tensor)
        arg0_tensor = torch.from_numpy(arg0)
        arg1_tensor = torch.from_numpy(arg1)
        arg2_tensor = torch.from_numpy(arg2)
        arg3_tensor = torch.from_numpy(arg3)
        arg4_tensor = torch.from_numpy(arg4)
        arg5_tensor = torch.from_numpy(arg5)
        arg6_tensor = torch.from_numpy(arg6)
        # Expected result using PyTorch
        conv1 = torch.nn.functional.conv2d(arg0_tensor, arg1_tensor, bias=arg2_tensor, stride=1, padding=0, dilation=1)
        tanh1 = torch.tanh(conv1)
        maxpool1 = torch.nn.functional.max_pool2d(tanh1, kernel_size=2, stride=2, padding=0)
        conv2 = torch.nn.functional.conv2d(maxpool1, arg3_tensor, bias=arg4_tensor, stride=1, padding=0, dilation=1)
        tanh2 = torch.tanh(conv2)
        maxpool2 = torch.nn.functional.max_pool2d(tanh2, kernel_size=2, stride=2, padding=0)
        reshaped = maxpool2.reshape(1,800)
        m=torch.nn.Linear(800, 500, bias=False)
        m.weight.data = arg5_tensor
        linear1=m(reshaped)
        tanh3 = torch.tanh(linear1)
        n=torch.nn.Linear(500, 10, bias=False)
        n.weight.data = arg6_tensor
        linear2=n(tanh3)
        tanh4 = torch.tanh(linear2)
        assert torch.allclose(res_tensor, tanh4, atol=1e-5), f"Expected {tanh4} but got {res_tensor}"

def test_forward():
    execute_forward()
    

if __name__ == "__main__":
    test_forward()

