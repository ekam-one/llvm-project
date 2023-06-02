set(LIBS
  MLIRMathToLLVM
  MLIRArithToLLVM
  MLIRAffineToStandard
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRMemRefToLLVM
  MLIRTransforms
  MLIRParser
  MLIRLLVMToLLVMIRTranslation
  MLIRFuncToLLVM
  MLIRTensorToLinalg
  MLIRBufferizationToMemRef
  MLIRTransforms
  MLIRAffineTransforms
  MhloToLinalg
  MLIRSCFToGPU
  MLIRGPUTransforms
  MLIRGPUToSPIRV
  MLIRTensorDialect
  MLIRSPIRVDialect
)

include_directories(${LLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR}/mhlo)

add_llvm_tool(lower-mlir
  lower-mlir.cpp
  DEPENDS
  ${LIBS}
)

target_link_libraries(lower-mlir PRIVATE ${LIBS})
llvm_update_compile_flags(lower-mlir)

mlir_check_all_link_libraries(lower-mlir)