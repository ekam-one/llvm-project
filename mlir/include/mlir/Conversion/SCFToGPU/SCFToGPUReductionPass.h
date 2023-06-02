//===- SCFToGPUReductionPass.h - Pass converting loops to GPU kernels ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_SCFTOGPU_SCFTOGPUREDUCTIONPASS_H_
#define MLIR_CONVERSION_SCFTOGPU_SCFTOGPUREDUCTIONPASS_H_

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTSCFTOGPUREDUCTION
#include "mlir/Conversion/Passes.h.inc"

/// Creates a pass that converts scf.for to gpu.reduce if reduction pattern gets
/// identified.
std::unique_ptr<Pass> createSCFLoopToGpuReductionPass();

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOGPU_SCFTOGPUPASS_H_
