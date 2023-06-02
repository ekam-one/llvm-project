//===- LowerTensorOps.cpp - Lower tensor operations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/None.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-tensor-ops"

namespace mlir {
#define GEN_PASS_DEF_LOWERTENSOROPS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

using namespace mlir::affine;

namespace {

class TensorTypeConverter : public TypeConverter {
public:
  TensorTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion([](TensorType type) {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
  }

  Optional<FunctionType> convertFunctionType(FunctionType type,
                                             SignatureConversion &result) {
    if (failed(convertSignatureArgs(type.getInputs(), result)))
      return std::nullopt;

    assert(type.getResults().empty());

    return FunctionType::get(type.getContext(), result.getConvertedTypes(), {});
  }

  bool isLegalType(FunctionType type) {
    if (type.getNumResults() != 0)
      return false;

    for (auto inTy : type.getInputs()) {
      if (inTy.isa<TensorType>())
        return false;
    }

    return true;
  }
};

template <typename T>
class TensorOpsConversionBase : public ConversionPattern {

protected:
  TensorTypeConverter &converter;

public:
  TensorOpsConversionBase(MLIRContext *context, TensorTypeConverter &_converter)
      : ConversionPattern(T::getOperationName(), 1, context),
        converter(_converter) {}
};

class FuncOpLowering : public TensorOpsConversionBase<func::FuncOp> {

public:
  using TensorOpsConversionBase<func::FuncOp>::TensorOpsConversionBase;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto func = cast<func::FuncOp>(op);

    auto funcType = func.getFunctionType();
    assert(funcType.getNumResults() == 0 &&
           "Fx to MLIR converter should append results are argument");

    TypeConverter::SignatureConversion sigConversion(funcType.getNumInputs());

    auto newFuncType = converter.convertFunctionType(funcType, sigConversion);
    assert(newFuncType.has_value());
    OpBuilder builder(op);
    auto newFunc = builder.create<func::FuncOp>(func.getLoc(), func.getName(),
                                                newFuncType.value());
    rewriter.inlineRegionBefore(func.getBody(), newFunc.getBody(),
                                newFunc.end());
    rewriter.applySignatureConversion(&newFunc.getBody(), sigConversion);

    rewriter.eraseOp(op);
    return success();
  }
};

class LinalgGenericLowering
    : public TensorOpsConversionBase<linalg::GenericOp> {

public:
  using TensorOpsConversionBase<linalg::GenericOp>::TensorOpsConversionBase;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto genOp = cast<linalg::GenericOp>(op);

    SmallVector<Value> newInputs, newInits;

    for (auto i = 0; i < operands.size(); i++) {
      if (i < genOp.getNumDpsInputs())
        newInputs.push_back(operands[i]);
      else
        newInits.push_back(operands[i]);
    }

    SmallVector<AffineMap> indexingMaps = genOp.getIndexingMapsArray();
    SmallVector<utils::IteratorType> iteratorTypes =
        genOp.getIteratorTypesArray();

    auto newOp = rewriter.create<linalg::GenericOp>(
        genOp.getLoc(), newInputs, newInits, indexingMaps, iteratorTypes);

    newOp.getRegion().takeBody(genOp->getRegion(0));
    rewriter.eraseOp(op);
    return success();
  }
};

class EmptyTensorLowering : public TensorOpsConversionBase<tensor::EmptyOp> {

public:
  using TensorOpsConversionBase<tensor::EmptyOp>::TensorOpsConversionBase;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto emptyOp = cast<tensor::EmptyOp>(op);

    auto tensorTy = emptyOp.getType();

    auto memrefTy = converter.convertType(tensorTy).dyn_cast<MemRefType>();
    assert(memrefTy);

    auto allocaOp = rewriter.create<memref::AllocaOp>(op->getLoc(), memrefTy);
    rewriter.replaceOp(op, allocaOp->getResults());
    return success();
  }
};

class TensorInsertSliceLowering
    : public TensorOpsConversionBase<tensor::InsertSliceOp> {

public:
  using TensorOpsConversionBase<tensor::InsertSliceOp>::TensorOpsConversionBase;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::errs() << "called 1\n";

    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);

    auto valueToStore = operands[0];

    auto loadValueToStore =
        rewriter.create<memref::LoadOp>(op->getLoc(), valueToStore);

    auto destination = operands[1];

    auto memrefOp = rewriter.create<memref::StoreOp>(
        op->getLoc(), loadValueToStore, destination,
        ValueRange({rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0)}));

    op->dropAllDefinedValueUses();
    op->dropAllUses();

    rewriter.eraseOp(op);

    return success();
  }
};

class ToMemRefLowering
    : public TensorOpsConversionBase<bufferization::ToMemrefOp> {

public:
  using TensorOpsConversionBase<
      bufferization::ToMemrefOp>::TensorOpsConversionBase;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    // TODO: Add required safety checks
    auto toMemRefOp = cast<bufferization::ToMemrefOp>(op);
    auto input = toMemRefOp.getOperand();
    rewriter.replaceOp(op, {input});
    return success();
  }
};

class TensorStoreLowering
    : public TensorOpsConversionBase<memref::TensorStoreOp> {

public:
  using TensorOpsConversionBase<memref::TensorStoreOp>::TensorOpsConversionBase;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto val = operands[0];
    auto ptr = operands[1];

    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, val, ptr);
    return success();
  }
};

template <typename T>
class BinaryElementwiseOpLowering : public TensorOpsConversionBase<T> {

public:
  using TensorOpsConversionBase<T>::TensorOpsConversionBase;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto sinOp = cast<T>(op);

    auto tensorTy = sinOp.getType().template dyn_cast<TensorType>();
    if (!tensorTy)
      return failure();

    SmallVector<int64_t, 2> lbs(tensorTy.getRank(), 0);
    SmallVector<int64_t, 2> steps(tensorTy.getRank(), 1);
    SmallVector<int64_t, 4> ubs;

    for (auto dim : tensorTy.getShape()) {
      ubs.push_back(dim);
    }

    auto resTy =
        this->converter.convertType(tensorTy).template dyn_cast<MemRefType>();
    assert(resTy);

    auto resultVar = rewriter.create<memref::AllocOp>(op->getLoc(), resTy);

    buildAffineLoopNest(
        rewriter, op->getLoc(), lbs, ubs, steps,
        ([&](OpBuilder &builder, Location loc, ValueRange inductionVars) {
          SmallVector<Value, 2> ivVector(inductionVars);
          auto input0 =
              builder.create<AffineLoadOp>(loc, operands[0], inductionVars);

          auto input1 =
              builder.create<AffineLoadOp>(loc, operands[1], inductionVars);

          auto binaryResult = builder.create<T>(loc, input0, input1);
          builder.create<AffineStoreOp>(loc, binaryResult, resultVar,
                                        inductionVars);
        }));

    rewriter.replaceOp(op, resultVar->getResults());
    return success();
  }
};

template <typename T>
class UnaryElementwiseOpLowering : public TensorOpsConversionBase<T> {

public:
  using TensorOpsConversionBase<T>::TensorOpsConversionBase;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto sinOp = cast<T>(op);

    auto tensorTy = sinOp.getType().template dyn_cast<TensorType>();
    if (!tensorTy)
      return failure();

    SmallVector<int64_t, 2> lbs(tensorTy.getRank(), 0);
    SmallVector<int64_t, 2> steps(tensorTy.getRank(), 1);
    SmallVector<int64_t, 4> ubs;

    for (auto dim : tensorTy.getShape()) {
      ubs.push_back(dim);
    }

    auto resTy =
        this->converter.convertType(tensorTy).template dyn_cast<MemRefType>();
    assert(resTy);

    auto resultVar = rewriter.create<memref::AllocOp>(op->getLoc(), resTy);

    buildAffineLoopNest(
        rewriter, op->getLoc(), lbs, ubs, steps,
        ([&](OpBuilder &builder, Location loc, ValueRange inductionVars) {
          SmallVector<Value, 2> ivVector(inductionVars);
          auto input =
              builder.create<AffineLoadOp>(loc, operands[0], inductionVars);

          auto unaryResult = builder.create<T>(loc, input);
          builder.create<AffineStoreOp>(loc, unaryResult, resultVar,
                                        inductionVars);
        }));

    rewriter.replaceOp(op, resultVar->getResults());
    return success();
  }
};

static void populateTensorLoweringPatterns(RewritePatternSet &patterns,
                                           TensorTypeConverter &converter,
                                           MLIRContext *context) {

  patterns.insert<FuncOpLowering, EmptyTensorLowering, TensorInsertSliceLowering,
                  LinalgGenericLowering /*, UnaryElementwiseOpLowering<math::SinOp>,
                  UnaryElementwiseOpLowering<math::CosOp>,
                  BinaryElementwiseOpLowering<arith::AddFOp>,
                  BinaryElementwiseOpLowering<arith::MulFOp>, ToMemRefLowering,
                  TensorStoreLowering*/>(context, converter);
}

static bool isLegalOp(Operation *op) {
  auto numOperands = op->getNumOperands();
  auto numResults = op->getNumResults();

  for (unsigned i = 0; i < numOperands; ++i) {
    auto operand = op->getOperand(i);
    if (operand.getType().isa<TensorType>())
      return false;
  }

  for (unsigned i = 0; i < numResults; ++i) {
    auto result = op->getResult(i);
    if (result.getType().isa<TensorType>())
      return false;
  }
  return true;
}

template <typename T>
static void addDynamicallyLegalOp(ConversionTarget &target) {
  target.addDynamicallyLegalOp<T>(
      [&](T op) { return isLegalOp(op.getOperation()); });
}

template <typename t1, typename t2, typename... tn>
static void addDynamicallyLegalOp(ConversionTarget &target) {
  addDynamicallyLegalOp<t1>(target);
  addDynamicallyLegalOp<t2, tn...>(target);
}

struct LowerTensorOps : public impl::LowerTensorOpsBase<LowerTensorOps> {

  void runOnOperation() {

    Operation *theModule = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    TensorTypeConverter converter;
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<AffineDialect, linalg::LinalgDialect,
                           arith::ArithDialect>();

    addDynamicallyLegalOp<math::SinOp, math::CosOp, arith::AddFOp,
                          arith::MulFOp, linalg::GenericOp>(target);

    target.addIllegalOp<bufferization::ToMemrefOp, memref::TensorStoreOp,
                        tensor::InsertSliceOp, tensor::EmptyOp>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp funcOp) {
      return converter.isLegalType(funcOp.getFunctionType());
    });

    populateTensorLoweringPatterns(patterns, converter,
                                   theModule->getContext());

    if (failed(applyPartialConversion(theModule, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createLowerTensorOps() {
  return std::make_unique<LowerTensorOps>();
}