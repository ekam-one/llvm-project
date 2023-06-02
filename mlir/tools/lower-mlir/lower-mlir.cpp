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
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUReductionPass.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
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

#include "IR/hlo_ops.h"
#include "transforms/passes.h"
#include "llvm/ADT/StringRef.h"
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
#include <optional>
#include <system_error>

using namespace llvm;
using namespace mlir;
using namespace mlir::affine;

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
  mlirPM.addPass(mlir::mhlo::createLegalizeHloToLinalgPass());
  mlirPM.addPass(mlir::createSimplifyLinalgCopyPass());
  mlirPM.addPass(mlir::createLowerTensorOps());

  mlirPM.addPass(mlir::createConvertLinalgToParallelLoopsPass());

  mlirPM.addPass(mlir::createGpuMapParallelLoopsPass());
  mlirPM.addPass(mlir::createParallelLoopToGpuPass());
  mlirPM.addPass(mlir::createSCFLoopToGpuReductionPass());
  mlirPM.addPass(createGpuKernelOutliningPass());
  mlirPM.addPass(mlir::createLowerAffinePass());
  mlirPM.addPass(mlir::createCanonicalizerPass());
  mlirPM.addPass(mlir::createConvertGPUToSPIRVPass(true));
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

std::string getMLIRTranslate() {
  auto mlirTranslate = llvm::sys::Process::GetEnv("MHLO_LLVM_RELEASE_BUILD");
  if (!mlirTranslate.has_value()) {
    llvm::errs()
        << "\n MLIR Translate Binary path MHLO_LLVM_RELEASE_BUILD is not set";
    return std::string();
  }
  return mlirTranslate.value() + "/bin/mlir-translate";
}

bool convertSPIRVDialectToBinary(std::string spirvMLIRFile) {
  auto ldCommand = getMLIRTranslate();
  llvm::SmallVector<llvm::StringRef, 16> argsArr = {ldCommand,
                                                    spirvMLIRFile,
                                                    "-serialize-spirv",
                                                    "-no-implicit-module",
                                                    "-o",
                                                    "/tmp/mhlo-output.spv"};

  std::string errorStr;
  bool execFailed = false;
  std::vector<llvm::Optional<StringRef>> redirects;
  redirects = {llvm::None, llvm::None, llvm::None};
  // FIXME: Replace with tpc llvm 12 when it is ready!!!
  int val = llvm::sys::ExecuteAndWait(ldCommand, argsArr, llvm::None, redirects,
                                      0, 0, &errorStr, &execFailed);

  if (execFailed || val != 0) {
    llvm::errs() << "\nERROR:mlir-translate tool execution failed : "
                 << errorStr;
    return 1;
  }

  return 0;
}

std::string getTargetRuntime() {
  auto deviceCompiler = llvm::sys::Process::GetEnv("DEVICE_COMPILER");
  if (!deviceCompiler.has_value()) {
    llvm::errs() << "\n DEVICE_COMPILER is not set";
    return std::string();
  }
  return deviceCompiler.value();
}

std::string getTargetInput() {
  auto input = llvm::sys::Process::GetEnv("DEVICE_INPUT_TYPE");
  if (!input.has_value()) {
    llvm::errs() << "\n DEVICE_INPUT_TYPE is not set";
    return std::string();
  }
  return input.value();
}

std::string getTargetArch() {
  auto arch = llvm::sys::Process::GetEnv("DEVICE_ARCH");
  if (!arch.has_value()) {
    llvm::errs() << "\n DEVICE_ARCH is not set";
    return std::string();
  }
  return arch.value();
}

std::string getTargetDevice() {
  auto device = llvm::sys::Process::GetEnv("TARGET_DEVICE");
  if (!device.has_value()) {
    llvm::errs() << "\n TARGET_DEVICE is not set";
    return std::string();
  }
  return device.value();
}

bool convertBinaryToVISA() {
  auto ldCommand = getTargetRuntime();
  auto targetInput = getTargetInput();
  auto device = getTargetDevice();
  auto arch = getTargetArch();
  llvm::SmallVector<llvm::StringRef, 16> argsArr = {
      ldCommand, targetInput, "-file", "/tmp/mhlo-output.spv", device, arch};

  std::string errorStr;
  bool execFailed = false;
  std::vector<llvm::Optional<StringRef>> redirects;
  redirects = {llvm::None, llvm::None, llvm::None};
  // FIXME: Replace with tpc llvm 12 when it is ready!!!
  int val = llvm::sys::ExecuteAndWait(ldCommand, argsArr, llvm::None, redirects,
                                      0, 0, &errorStr, &execFailed);

  if (execFailed || val != 0) {
    llvm::errs() << "\nERROR:VISA conversion tool execution failed : "
                 << errorStr;
    return 1;
  }

  return 0;
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
                  bufferization::BufferizationDialect, mhlo::MhloDialect,
                  linalg::LinalgDialect, gpu::GPUDialect, memref::MemRefDialect,
                  tensor::TensorDialect, scf::SCFDialect>();

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

  SmallVector<std::string> spirvIRFiles;
  unsigned i = 1;
  mod.get()->walk([&](mlir::spirv::ModuleOp spirvModule) {
    std::error_code EC;
    auto file = "/tmp/tmp-mho_spirv_" + std::to_string(i) + ".mlir";
    raw_fd_stream spirvMLIRFile(file, EC);
    spirvModule.print(spirvMLIRFile);
    spirvIRFiles.push_back(file);
    spirvMLIRFile.close();
    i++;
  });

  for (auto sprivMLIRModule : spirvIRFiles) {
    llvm::errs() << "Lowering " << sprivMLIRModule << "\n";
    if (auto returnVal = convertSPIRVDialectToBinary(sprivMLIRModule))
      return returnVal;

    llvm::errs() << "SPIRV Binary emitted\n";

    if (auto returnVal = convertBinaryToVISA())
      return returnVal;

    llvm::errs() << "VISA Emitted\n";
  }

  return 0;
}