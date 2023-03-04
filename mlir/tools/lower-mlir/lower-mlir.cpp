//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <system_error>

using namespace llvm;
using namespace mlir;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("Input mlir file"), cl::init("-"));

static cl::opt<std::string>
    outputLibName("o", cl::desc("Name for the library to be generated"),
                  cl::init("/tmp/out.so"));

static cl::opt<bool> enableIRPrinting("print-mlir-after-all",
                                      cl::desc("Print IR after each pass"),
                                      cl::init(false));

static cl::opt<bool>
    lowerTorchOps("lower-torch-ops",
                  cl::desc("Lower the torch dialect to affine"),
                  cl::init(false));

static OwningOpRef<mlir::Operation *> loadModule(std::string input,
                                                 MLIRContext *context) {
  std::string modName = "torch-module";
  if (input.empty()) {
    llvm::errs() << "\n[LOWER-MLIR]: MLIR is empty";
    return nullptr;
  }

  SourceMgr sourceMgr;
  SourceMgr sourceMgrForDiagnostic;
  std::string ignored;
  auto buff = llvm::MemoryBuffer::getMemBufferCopy(input, modName);
  sourceMgr.AddNewSourceBuffer(std::move(buff), SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgrForDiagnostic, context);

  OwningOpRef<mlir::Operation *> mod(parseSourceFile(sourceMgr, context));
  if (!mod) {
    llvm::errs() << "\n[LOWER-MLIR]: Failed to parse the MLIR Module";
    return nullptr;
  }

  return mod;
}

static void addMLIRLoweringPipeline(mlir::PassManager &mlirPM) {

  if (lowerTorchOps) {
    mlirPM.addPass(mlir::createLowerTensorOps());
    mlirPM.addPass(mlir::createLoopFusionPass());
  }

  mlirPM.addPass(mlir::createLowerAffinePass());
  mlirPM.addPass(mlir::createConvertSCFToCFPass());
  mlirPM.addPass(mlir::createConvertMathToLLVMPass());
  // mlirPM.addPass(mlir::arith::createConvertArithmeticToLLVMPass());
  mlirPM.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

  mlir::ConvertFuncToLLVMPassOptions opts;
  opts.useBarePtrCallConv = true;
  // mlirPM.addPass(mlir::createLowerToLLVMPass(opts));
  mlirPM.addPass(mlir::createConvertControlFlowToLLVMPass());
  mlirPM.addPass(mlir::createArithToLLVMConversionPass());
  mlirPM.addPass(mlir::createConvertFuncToLLVMPass(opts));
  mlirPM.addPass(mlir::createCSEPass());
  mlirPM.addPass(mlir::createCanonicalizerPass());
  mlirPM.addPass(mlir::createReconcileUnrealizedCastsPass());
}

static bool lowerMLIRToShared(mlir::OwningOpRef<mlir::Operation *> &mod,
                              std::string filename) {
  std::error_code error;
  if (error.value()) {
    llvm::errs() << error.message() << "\n";
    return false;
  }

  llvm::LLVMContext llvmContext;
  mlir::registerLLVMDialectTranslation(*mod.get()->getContext());
  auto llvmModule =
      translateModuleToLLVMIR(mod.get(), llvmContext, "MLIRModule");
  if (!llvmModule) {
    llvm::errs() << "\n[LOWER-MLIR]: Failed to translate MLIR to LLVM-IR";
    return false;
  }

  auto tempFile = llvm::sys::path::filename(StringRef(filename));
  std::string tempFilename = "/tmp/" + tempFile.str() + "%%%%%%%.ll";
  auto tmpFile = llvm::sys::fs::TempFile::create(tempFilename);
  if (!tmpFile) {
    llvm::errs() << "\n[LOWER-MLIR]: Failed to create temporary file";
    return false;
  }

  std::error_code ec;
  llvm::raw_fd_ostream tos(tmpFile.get().TmpName, ec, llvm::sys::fs::OF_None);
  llvmModule->print(tos, nullptr);
  tos.flush();
  tos.close();

  // Look for clang-17 only
  auto compiler = "clang-17";
  static ErrorOr<std::string> compilerPath = sys::findProgramByName(compiler);
  if (!compilerPath) {
    errs() << "Unable to find compiler absolute path.";
    return 1;
  }

  llvm::SmallVector<llvm::StringRef, 16> argsArr = {
      compiler,       "-c",
      "-O3",          "-fPIC",
      "-Wall",        "-Wno-unused-variable",
      "-ffast-math",  "-fno-finite-math-only",
      "-march=native"};

  argsArr.push_back(tmpFile.get().TmpName);
  argsArr.push_back("-o");
  auto objectFileName = tmpFile.get().TmpName;
  objectFileName = objectFileName.substr(0, objectFileName.size() - 2) + "o";
  argsArr.push_back(objectFileName);

  std::string errorString;
  bool execFailed = false;

  /// Generate object file
  int val = llvm::sys::ExecuteAndWait(compilerPath.get(), argsArr, {}, {}, 0, 0,
                                      &errorString, &execFailed);

  if (auto e = tmpFile->discard()) {
    llvm::errs() << "\n[LOWER-MLIR]: Temp file generation failed";
    return false;
  }

  if (execFailed || val != 0) {
    llvm::errs() << "\n[LOWER-MLIR]: Generating -fPIC code failed : "
                 << errorString;
    return false;
  }

  argsArr.clear();
  errorString.clear();

  // Generate so from *.o
  argsArr.push_back(compiler);
  argsArr.push_back("-shared");
  argsArr.push_back(objectFileName);
  argsArr.push_back("-o");
  argsArr.push_back(filename);

  val = llvm::sys::ExecuteAndWait(compilerPath.get(), argsArr, {}, {}, 0, 0,
                                  &errorString, &execFailed);

  if (execFailed || val != 0) {
    llvm::errs() << "\n[LOWER-MLIR]: Generating -shared library failed : "
                 << errorString;
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  std::string errMsg;
  auto file = openInputFile(inputFilename, &errMsg);
  if (!file) {
    llvm::errs() << errMsg << "\n";
    return 1;
  }

  DialectRegistry registry;
  registry.insert<AffineDialect, math::MathDialect, LLVM::LLVMDialect,
                  cf::ControlFlowDialect, func::FuncDialect,
                  bufferization::BufferizationDialect>();

  MLIRContext context(registry);

  context.allowUnregisteredDialects();
  context.disableMultithreading();

  std::error_code error;
  std::string filename = outputLibName.c_str();
  if (error.value()) {
    llvm::errs() << error.message() << "\n";
    return 1;
  }

  OwningOpRef<mlir::Operation *> mod = nullptr;
  std::string input = file->getBuffer().str();

  assert(!input.empty());
  mod = loadModule(input, &context);

  if (!mod) {
    llvm::errs() << "\n[LOWER-MLIR]: Failed to parse the MLIR Module";
    return 1;
  }

  mlir::PassManager mlirPM(&context);
  mlirPM.enableVerifier(true);

  if (enableIRPrinting) {
    mlirPM.enableIRPrinting();
  }
  addMLIRLoweringPipeline(mlirPM);

  if (failed(mlirPM.run(mod.get()))) {
    llvm::errs() << "Failed to run MLIR pass manager\n";
    return 1;
  }

  if (!lowerMLIRToShared(mod, filename))
    return 1;

  return 0;
}