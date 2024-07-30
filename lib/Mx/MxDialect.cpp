// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Mx/MxDialect.h"
#include "Mx/MxOps.h"

using namespace mlir;
using namespace mx;

//===----------------------------------------------------------------------===//
// mx dialect.
//===----------------------------------------------------------------------===//

#include "Mx/MxOpsDialect.cpp.inc"

void MxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mx/MxOps.cpp.inc"
      >();
}

void mx::ConstantOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  mx::ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::Operation *MxDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  return builder.create<mx::ConstantOp>(
      loc, type, llvm::cast<mlir::DenseElementsAttr>(value));
}

//Add Op
void mx::AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                mlir::Value lhs, mlir::Value rhs) {
  auto dataType=llvm::dyn_cast<UnrankedTensorType>(lhs.getType());
  state.addTypes(dataType);
  state.addOperands({lhs, rhs});
}

void mx::MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                mlir::Value lhs, mlir::Value rhs) {
  auto dataType=llvm::dyn_cast<UnrankedTensorType>(lhs.getType());
  state.addTypes(dataType);
  state.addOperands({lhs, rhs});
}

void mx::AddMulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                mlir::Value first, mlir::Value second, mlir::Value third) {
  auto dataType=llvm::dyn_cast<UnrankedTensorType>(first.getType());
  state.addTypes(dataType);
  state.addOperands({first, second, third});
}