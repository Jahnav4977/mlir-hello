cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()


if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(mlir-hello LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(AddMLIRPython)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(hello-opt)

#python bindings 
if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  # If parent projects want to configure where to place the python packages,
  # respect that.
  if(NOT TORCH_MLIR_PYTHON_PACKAGES_DIR)
    set(TORCH_MLIR_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_packages")
  endif()
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  declare_mlir_python_sources(HelloMLIRPythonHelloExtensionsSources)
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

message(STATUS "TORCH_MLIR_PYTHON_PACKAGES_DIR is set to: ${TORCH_MLIR_PYTHON_PACKAGES_DIR}")