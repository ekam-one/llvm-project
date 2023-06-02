//===- SCFToGPUPass.cpp - Convert a loop nest to a GPU kernel -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToGPU/SCFToGPUReductionPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"
#include <tuple>

namespace mlir {
#define GEN_PASS_DEF_CONVERTSCFTOGPUREDUCTION
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

enum ReductionKind { Add, Max, Min, Unknown };

struct ReductionLoop {
  scf::ForOp loop;
  ReductionKind kind;
  Operation *reductionOp = nullptr;
  int incomingOpPosToReduce = -1;
};

enum MappingLevel { MapGrid = 0, MapBlock = 1, Sequential = 2 };

static constexpr int kNumHardwareIds = 3;

/// Bounded increment on MappingLevel. Increments to the next
/// level unless Sequential was already reached.
static MappingLevel &operator++(MappingLevel &mappingLevel) {
  if (mappingLevel < Sequential) {
    mappingLevel = static_cast<MappingLevel>(mappingLevel + 1);
  }
  return mappingLevel;
}

/// Computed the hardware id to use for a given mapping level. Will
/// assign x,y and z hardware ids for the first 3 dimensions and use
/// sequential after.
/// TODO: Make this use x for the inner-most loop that is
/// distributed to map to x, the next innermost to y and the next innermost to
/// z.
static mlir::gpu::Processor getHardwareIdForMapping(MappingLevel level,
                                                    int dimension) {

  if (dimension >= kNumHardwareIds || level == Sequential)
    return mlir::gpu::Processor::Sequential;
  switch (level) {
  case MapGrid:
    switch (dimension) {
    case 0:
      return mlir::gpu::Processor::BlockX;
    case 1:
      return mlir::gpu::Processor::BlockY;
    case 2:
      return mlir::gpu::Processor::BlockZ;
    default:
      return mlir::gpu::Processor::Sequential;
    }
    break;
  case MapBlock:
    switch (dimension) {
    case 0:
      return mlir::gpu::Processor::ThreadX;
    case 1:
      return mlir::gpu::Processor::ThreadY;
    case 2:
      return mlir::gpu::Processor::ThreadZ;
    default:
      return mlir::gpu::Processor::Sequential;
    }
  default:;
  }
  return mlir::gpu::Processor::Sequential;
}

StringRef getMappingAttrName() { return "mapping"; }

LogicalResult
setMappingAttr(scf::ForOp ploopOp,
               ArrayRef<mlir::gpu::ParallelLoopDimMappingAttr> mapping) {
  // Verify that each processor is mapped to only once.
  llvm::DenseSet<gpu::Processor> specifiedMappings;
  for (auto dimAttr : mapping) {
    gpu::Processor processor = dimAttr.getProcessor();
    if (processor != gpu::Processor::Sequential &&
        specifiedMappings.count(processor))
      return ploopOp.emitError(
          "invalid mapping multiple loops to same processor");
  }
  ArrayRef<Attribute> mappingAsAttrs(mapping.data(), mapping.size());
  ploopOp->setAttr(getMappingAttrName(),
                   ArrayAttr::get(ploopOp.getContext(), mappingAsAttrs));
  return success();
}

/// Add mapping information to the given parallel loop. Do not add
/// mapping information if the loop already has it. Also, don't
/// start a mapping at a nested loop.
static void mapParallelOp(scf::ForOp forOp,
                          MappingLevel mappingLevel = MapGrid) {
  // Do not try to add a mapping to already mapped loops or nested loops.
  if (forOp->getAttr(getMappingAttrName()) ||
      ((mappingLevel == MapGrid) && forOp->getParentOfType<scf::ForOp>()))
    return;

  MLIRContext *ctx = forOp.getContext();
  Builder b(ctx);
  SmallVector<gpu::ParallelLoopDimMappingAttr, 4> attrs;
  attrs.reserve(1);

  attrs.push_back(b.getAttr<gpu::ParallelLoopDimMappingAttr>(
      getHardwareIdForMapping(mappingLevel, 0), b.getDimIdentityMap(),
      b.getDimIdentityMap()));

  (void)setMappingAttr(forOp, attrs);
}

static bool isMappedToProcessor(gpu::Processor processor) {
  return processor != gpu::Processor::Sequential;
}

static unsigned getLaunchOpArgumentNum(gpu::Processor processor) {
  switch (processor) {
  case gpu::Processor::BlockX:
    return 0;
  case gpu::Processor::BlockY:
    return 1;
  case gpu::Processor::BlockZ:
    return 2;
  case gpu::Processor::ThreadX:
    return 3;
  case gpu::Processor::ThreadY:
    return 4;
  case gpu::Processor::ThreadZ:
    return 5;
  default:;
  }
  llvm_unreachable(
      "invalid processor type while retrieving launch op argument number");
}

/// Tries to derive a static upper bound from the defining operation of
/// `upperBound`.
static Value deriveStaticUpperBound(Value upperBound, OpBuilder &rewriter) {
  if (auto op = upperBound.getDefiningOp<arith::ConstantIndexOp>()) {
    return op;
  }

  if (auto minOp = upperBound.getDefiningOp<mlir::affine::AffineMinOp>()) {
    for (const AffineExpr &result : minOp.getMap().getResults()) {
      if (auto constExpr = result.dyn_cast<AffineConstantExpr>()) {
        return rewriter.create<arith::ConstantIndexOp>(minOp.getLoc(),
                                                       constExpr.getValue());
      }
    }
  }

  if (auto minOp = upperBound.getDefiningOp<arith::MinSIOp>()) {
    for (Value operand : {minOp.getLhs(), minOp.getRhs()}) {
      if (auto staticBound = deriveStaticUpperBound(operand, rewriter))
        return staticBound;
    }
  }

  if (auto multiplyOp = upperBound.getDefiningOp<arith::MulIOp>()) {
    if (auto lhs = dyn_cast_or_null<arith::ConstantIndexOp>(
            deriveStaticUpperBound(multiplyOp.getOperand(0), rewriter)
                .getDefiningOp()))
      if (auto rhs = dyn_cast_or_null<arith::ConstantIndexOp>(
              deriveStaticUpperBound(multiplyOp.getOperand(1), rewriter)
                  .getDefiningOp())) {
        // Assumptions about the upper bound of minimum computations no longer
        // work if multiplied by mixed signs, so abort in this case.
        if ((lhs.value() < 0) != (rhs.value() < 0))
          return {};

        return rewriter.create<arith::ConstantIndexOp>(
            multiplyOp.getLoc(), lhs.value() * rhs.value());
      }
  }

  return {};
}

static LogicalResult processReductionLoop(
    scf::ForOp parallelOp, gpu::LaunchOp launchOp, IRMapping &cloningMap,
    SmallVectorImpl<Operation *> &worklist,
    DenseMap<gpu::Processor, Value> &bounds, OpBuilder &rewriter) {
  // TODO: Verify that this is a valid GPU mapping.
  // processor ids: 0-2 block [x/y/z], 3-5 -> thread [x/y/z], 6-> sequential
  ArrayAttr mapping =
      parallelOp->getAttrOfType<ArrayAttr>(getMappingAttrName());

  // TODO: Support reductions.
  if (!mapping)
    return failure();

  Location loc = parallelOp.getLoc();

  auto launchIndependent = [&launchOp](Value val) {
    return val.getParentRegion()->isAncestor(launchOp->getParentRegion());
  };

  auto ensureLaunchIndependent = [&rewriter,
                                  launchIndependent](Value val) -> Value {
    if (launchIndependent(val))
      return val;
    if (auto constOp = val.getDefiningOp<arith::ConstantOp>())
      return rewriter.create<arith::ConstantOp>(constOp.getLoc(),
                                                constOp.getValue());
    return {};
  };

  for (auto config :
       llvm::zip(mapping, ValueRange({parallelOp.getInductionVar()}),
                 ValueRange({parallelOp.getLowerBound()}),
                 ValueRange({parallelOp.getUpperBound()}),
                 ValueRange({parallelOp.getStep()}))) {
    Attribute mappingAttribute;
    Value iv, lowerBound, upperBound, step;
    std::tie(mappingAttribute, iv, lowerBound, upperBound, step) = config;
    auto annotation =
        dyn_cast<gpu::ParallelLoopDimMappingAttr>(mappingAttribute);
    if (!annotation)
      return parallelOp.emitOpError()
             << "expected mapping attribute for lowering to GPU";
    Value newIndex;
    gpu::Processor processor = annotation.getProcessor();

    if (isMappedToProcessor(processor)) {
      // Use the corresponding thread/grid index as replacement for the loop iv.
      Value operand =
          launchOp.getBody().getArgument(getLaunchOpArgumentNum(processor));
      // Take the indexmap and add the lower bound and step computations in.
      // This computes operand * step + lowerBound.
      // Use an affine map here so that it composes nicely with the provided
      // annotation.
      AffineMap lowerAndStep = AffineMap::get(
          1, 2,
          rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(0) +
              rewriter.getAffineSymbolExpr(1));
      newIndex = rewriter.create<mlir::affine::AffineApplyOp>(
          loc, annotation.getMap().compose(lowerAndStep),
          ValueRange{operand, step, lowerBound});
      // If there was also a bound, insert that, too.
      // TODO: Check that we do not assign bounds twice.
      if (annotation.getBound()) {
        // We pass as the single operand to the bound-map the number of
        // iterations, which is (upperBound - lowerBound) ceilDiv step. To
        // support inner loops with dynamic upper bounds (as generated by e.g.
        // tiling), try to derive a max for the bounds. If the used bound for
        // the hardware id is imprecise, wrap the contained code into a
        // conditional. If the lower-bound is constant or defined before the
        // launch, we can use it in the launch bounds. Otherwise fail.
        if (!launchIndependent(lowerBound) &&
            !isa_and_nonnull<arith::ConstantOp>(lowerBound.getDefiningOp()))
          return failure();
        // The step must also be constant or defined outside of the loop nest.
        if (!launchIndependent(step) &&
            !isa_and_nonnull<arith::ConstantOp>(step.getDefiningOp()))
          return failure();
        // If the upper-bound is constant or defined before the launch, we can
        // use it in the launch bounds directly. Otherwise try derive a bound.
        bool boundIsPrecise =
            launchIndependent(upperBound) ||
            isa_and_nonnull<arith::ConstantOp>(upperBound.getDefiningOp());
        {
          PatternRewriter::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(launchOp);
          if (!boundIsPrecise) {
            upperBound = deriveStaticUpperBound(upperBound, rewriter);
            if (!upperBound) {
              llvm_unreachable("not handled");
            }
          }
          // Compute the number of iterations needed. We compute this as an
          // affine expression ceilDiv (upperBound - lowerBound) step. We use
          // affine.apply here so that it composes nicely with the provided map.
          AffineMap stepMap = AffineMap::get(
              1, 2,
              ((rewriter.getAffineDimExpr(0) - rewriter.getAffineSymbolExpr(0))
                   .ceilDiv(rewriter.getAffineSymbolExpr(1))));
          Value launchBound = rewriter.create<mlir::affine::AffineApplyOp>(
              loc, annotation.getBound().compose(stepMap),
              ValueRange{
                  ensureLaunchIndependent(
                      cloningMap.lookupOrDefault(upperBound)),
                  ensureLaunchIndependent(
                      cloningMap.lookupOrDefault(lowerBound)),
                  ensureLaunchIndependent(cloningMap.lookupOrDefault(step))});
          // todo(herhut,ravishankarm): Update the behavior of setMappingAttr
          // when this condition is relaxed.
          if (bounds.contains(processor)) {
            llvm_unreachable("cannot redefine the bound for processor " +
                             Twine(static_cast<int64_t>(processor)));
          }
          bounds[processor] = launchBound;
        }
        if (!boundIsPrecise) {
          // We are using an approximation, create a surrounding conditional.
          Value originalBound = std::get<3>(config);
          arith::CmpIOp pred = rewriter.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, newIndex,
              cloningMap.lookupOrDefault(originalBound));
          scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, pred, false);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
          // Put a sentinel into the worklist so we know when to pop out of the
          // if body again. We use the launchOp here, as that cannot be part of
          // the bodies instruction.
          worklist.push_back(launchOp.getOperation());
        }
      }
    } else {
      // Create a sequential for loop.
      auto loopOp = rewriter.create<scf::ForOp>(
          loc, cloningMap.lookupOrDefault(lowerBound),
          cloningMap.lookupOrDefault(upperBound),
          cloningMap.lookupOrDefault(step));
      newIndex = loopOp.getInductionVar();
      rewriter.setInsertionPointToStart(loopOp.getBody());
      // Put a sentinel into the worklist so we know when to pop out of the loop
      // body again. We use the launchOp here, as that cannot be part of the
      // bodies instruction.
      worklist.push_back(launchOp.getOperation());
    }
    cloningMap.map(iv, newIndex);
  }

  // Propagate custom user defined optional attributes, that can be used at
  // later stage, such as extension data for GPU kernel dispatch
  for (const auto &namedAttr : parallelOp->getAttrs()) {
    if (namedAttr.getName() == getMappingAttrName())
      continue;
    launchOp->setAttr(namedAttr.getName(), namedAttr.getValue());
  }

  Block *body = parallelOp.getBody();
  worklist.reserve(worklist.size() + body->getOperations().size());
  for (Operation &op : llvm::reverse(body->without_terminator()))
    worklist.push_back(&op);
  return success();
}

struct SCFToGpuReduction
    : public impl::ConvertSCFToGpuReductionBase<SCFToGpuReduction> {

  // Function will be run only after isReductionLoop is run on scf::ForOp,
  // therefore we are assuming the body has single store & value to be stored to
  // the memref::store decides our reduction kind.
  // TODO: need to be generalized.
  std::tuple<ReductionKind, Operation *, int>
  collectReductionDetails(scf::ForOp loop) {
    SmallVector<memref::StoreOp> storeOps;
    loop.walk([&](memref::StoreOp store) { storeOps.push_back(store); });

    if (storeOps.size() != 1)
      return std::make_tuple<ReductionKind, Operation *>(
          Unknown, loop.getOperation(),
          -1); // TODO: add analysis for multiple
               // reduction types in a single loop.

    auto destAddr = storeOps[0].getMemRef();

    memref::LoadOp sourceLoad;

    loop.walk([&](memref::LoadOp load) {
      if (load.getMemRef() == destAddr) {
        sourceLoad = load;
      }
    });

    auto valueToStore = storeOps[0].getValueToStore();

    auto valueFrom = valueToStore.getDefiningOp();

    int incomingRedValuePosition = -1;
    for (unsigned i = 0; i < valueFrom->getOperands().size(); i++) {
      if (valueFrom->getOperands()[i] != sourceLoad)
        incomingRedValuePosition = i;
    }

    if (valueToStore.getDefiningOp<arith::AddFOp>())
      return std::make_tuple<ReductionKind, Operation *>(
          Add, valueToStore.getDefiningOp(), incomingRedValuePosition);

    if (valueToStore.getDefiningOp<arith::MaxFOp>())
      return std::make_tuple<ReductionKind, Operation *>(
          Max, valueToStore.getDefiningOp(), incomingRedValuePosition);

    if (valueToStore.getDefiningOp<arith::MinFOp>())
      return std::make_tuple<ReductionKind, Operation *>(
          Min, valueToStore.getDefiningOp(), incomingRedValuePosition);

    return std::make_tuple<ReductionKind, Operation *>(
        Unknown, loop.getOperation(), incomingRedValuePosition);
  }

  bool isReductionLoop(scf::ForOp loop) {
    SmallVector<memref::StoreOp> storeOps;
    loop.walk([&](memref::StoreOp store) { storeOps.push_back(store); });

    if (storeOps.size() != 1)
      return false; // TODO: add analysis for multiple reduction types in a
                    // single loop.

    auto storeOp = storeOps[0];
    auto destAddr = storeOp.getMemRef();

    memref::LoadOp sourceLoad;

    loop.walk([&](memref::LoadOp load) {
      if (load.getMemRef() == destAddr) {
        sourceLoad = load;
      }
    });

    auto valueToStore = storeOp.getValueToStore();
    auto storeIndices = storeOp.getIndices();

    auto valueFrom = valueToStore.getDefiningOp();

    for (auto operand : valueFrom->getOperands()) {
      if (operand == sourceLoad)
        return true;
    }

    return false;
  }

  void runOnOperation() override {
    SmallVector<ReductionLoop> reductionLoops;
    getOperation()->walk([&](scf::ForOp forOp) {
      if (isReductionLoop(forOp)) {
        auto redInfo = collectReductionDetails(forOp);
        ReductionLoop candidate;
        candidate.loop = forOp;
        candidate.kind = std::get<0>(redInfo);
        candidate.reductionOp = std::get<1>(redInfo);
        candidate.incomingOpPosToReduce = std::get<2>(redInfo);
        reductionLoops.push_back(candidate);
      }
    });

    for (auto candidate : reductionLoops) {
      auto loop = candidate.loop;
      OpBuilder builder(loop);
      Value constantOne =
          builder.create<arith::ConstantIndexOp>(loop.getLoc(), 1);
      gpu::LaunchOp launchOp = builder.create<gpu::LaunchOp>(
          loop.getLoc(), constantOne, constantOne, constantOne, constantOne,
          constantOne, constantOne);

      builder.setInsertionPointToEnd(&launchOp.getBody().front());
      builder.create<gpu::TerminatorOp>(loop.getLoc());
      builder.setInsertionPointToStart(&launchOp.getBody().front());

      mapParallelOp(loop);

      IRMapping cloningMap;
      llvm::DenseMap<gpu::Processor, Value> launchBounds;
      SmallVector<Operation *, 16> worklist;
      if (failed(processReductionLoop(loop, launchOp, cloningMap, worklist,
                                      launchBounds, builder)))
        return;

      // Whether we have seen any side-effects. Reset when leaving an inner
      // scope.
      bool seenSideeffects = false;
      // Whether we have left a nesting scope (and hence are no longer
      // innermost).
      bool leftNestingScope = false;
      while (!worklist.empty()) {
        Operation *op = worklist.pop_back_val();

        Operation *clone = nullptr;
        if (candidate.reductionOp == op) {
          auto valueToReduce = cloningMap.lookupOrNull(
              op->getOperand(candidate.incomingOpPosToReduce));

          mlir::gpu::AllReduceOperationAttr reductionKind;

          switch (candidate.kind) {
          case Add:
            reductionKind = mlir::gpu::AllReduceOperationAttr::get(
                op->getContext(), mlir::gpu::AllReduceOperation::ADD);
            break;
          case Max:
            reductionKind = mlir::gpu::AllReduceOperationAttr::get(
                op->getContext(), mlir::gpu::AllReduceOperation::MAX);
            break;
          case Min:
            reductionKind = mlir::gpu::AllReduceOperationAttr::get(
                op->getContext(), mlir::gpu::AllReduceOperation::MIN);
            break;
          case Unknown:
            llvm_unreachable("unhandled red type");
          }

          clone = builder.create<gpu::AllReduceOp>(
              op->getLoc(), valueToReduce, reductionKind,
              UnitAttr::get(op->getContext()));
          cloningMap.map(op->getResults(), clone->getResults());
        } else {
          clone = builder.clone(*op, cloningMap);
        }
        cloningMap.map(op->getResults(), clone->getResults());
        // Check for side effects.
        // TODO: Handle region side effects properly.
        seenSideeffects |=
            !isMemoryEffectFree(clone) || clone->getNumRegions() != 0;
        // If we are no longer in the innermost scope, sideeffects are
        // disallowed.
        if (seenSideeffects && leftNestingScope)
          return;
      }

      // Now that we succeeded creating the launch operation, also update the
      // bounds.
      for (auto bound : launchBounds)
        launchOp.setOperand(getLaunchOpArgumentNum(std::get<0>(bound)),
                            std::get<1>(bound));
    }

    for (auto i = 0; i < reductionLoops.size(); i++) {
      reductionLoops[i].loop.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createSCFLoopToGpuReductionPass() {
  return std::make_unique<SCFToGpuReduction>();
}
