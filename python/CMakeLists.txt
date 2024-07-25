set(HELLO_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hello_mlir")

# We vendor our own MLIR instance in the `torch_mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=hello_mlir.")

declare_mlir_python_sources(HelloMLIRPythonSources)
declare_mlir_python_sources(HelloMLIRPythonExtensions)

declare_mlir_python_sources(HelloMLIRPythonSources.Dialects
  ROOT_DIR "${HELLO_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT HelloMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT HelloMLIRPythonSources.Dialects
  ROOT_DIR "${HELLO_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE HelloBinding.td
  SOURCES __init__.py
  DIALECT_NAME hello
)

add_mlir_python_modules(HelloMLIRPythonModules
  ROOT_PREFIX "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/hello_mlir/hello_mlir"
  INSTALL_PREFIX "python_packages/hello_mlir/hello_mlir"
  DECLARED_SOURCES ${HelloMLIRPythonSources}
  COMMON_CAPI_LINK_LIBS
    HelloMLIRAggregateCAPI
)