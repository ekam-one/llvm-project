//===- SimplifyLinalgCopy.cpp - Lower tensor operations
//-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-tensor-ops"

namespace mlir {
#define GEN_PASS_DEF_SIMPLIFYLINALGCOPY
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

struct SimplifyLinalgCopy
    : public impl::SimplifyLinalgCopyBase<SimplifyLinalgCopy> {

  // Ideally this should be a pattern match, but tensor operations do not have
  // memory effects which gets them erased by internal canonicalizer running in
  // full conversions.
  void runOnOperation() {

    auto funcOp = getOperation();

    SmallVector<tensor::InsertSliceOp> copyOps;
    SmallVector<tensor::EmptyOp> emptyOps;
    DenseMap<Value, Value> emptyTensorToBlockArgMap;

    funcOp->walk([&](tensor::InsertSliceOp copy) {
      if (auto genericOp =
              copy.getSource().getDefiningOp<linalg::GenericOp>()) {
        if (auto emptyTensorOp =
                genericOp.getOutputs()[0].getDefiningOp<tensor::EmptyOp>()) {
          if (copy.getDest().isa<BlockArgument>()) {
            copyOps.push_back(copy);
            emptyOps.push_back(emptyTensorOp);
            emptyTensorToBlockArgMap[emptyTensorOp] = copy.getDest();
          }
        }
      }
    });

    for (auto vmap : emptyTensorToBlockArgMap) {
      vmap.first.replaceAllUsesWith(vmap.second);
    }

    assert(copyOps.size() == emptyOps.size());

    for (unsigned i = 0; i < copyOps.size(); i++) {
      copyOps[i].getOperation()->erase();
      emptyOps[i].getOperation()->erase();
    }

    SmallVector<linalg::GenericOp> genericOpsToUpdate;
    funcOp->walk([&](linalg::GenericOp genOp) {
      if (genOp.use_empty()) {
        genericOpsToUpdate.push_back(genOp);
      }
    });

    for (unsigned i = 0; i < genericOpsToUpdate.size(); i++) {
      auto genericOp = genericOpsToUpdate[i];
      OpBuilder builder(genericOp);
      SmallVector<Value> inputsAndInits;

      ValueRange inputs = genericOp.getInputs();
      ValueRange inits = genericOp.getOutputs();

      SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
      SmallVector<utils::IteratorType> iteratorTypes =
          genericOp.getIteratorTypesArray();

      auto packedLinalgOp = builder.create<linalg::GenericOp>(
          genericOp.getLoc(), inits.getTypes(), inputs, inits, indexingMaps,
          iteratorTypes);
      packedLinalgOp.getRegion().takeBody(genericOp->getRegion(0));

      genericOp.erase();
    }

    DenseMap<tensor::FromElementsOp, tensor::ExtractOp>
        constructAndExtractTensorMap;

    funcOp->walk([&](tensor::ExtractOp extract) {
      if (extract.getIndices().empty()) {
        if (auto fromElementsOp =
                extract.getTensor().getDefiningOp<tensor::FromElementsOp>()) {
          assert(fromElementsOp.getElements().size() == 1);
          constructAndExtractTensorMap[fromElementsOp] = extract;
        }
      }
    });

    for (auto itr : constructAndExtractTensorMap) {
      auto element = itr.getFirst().getElements()[0];
      itr.getSecond()->getResult(0).replaceAllUsesWith(element);
      itr.getSecond().erase();
      itr.getFirst()->erase();
    }

    SmallVector<linalg::FillOp> fillOpsToErase;
    funcOp->walk([&](linalg::FillOp fillOp) {
      if (fillOp.getInputs().size() == 1) {
        if (auto constOp =
                fillOp.getInputs()[0].getDefiningOp<arith::ConstantOp>()) {
          if (fillOp.getOutputs().size() == 1) {
            if (auto emptyTensor =
                    fillOp.getOutputs()[0].getDefiningOp<tensor::EmptyOp>()) {
              fillOpsToErase.push_back(fillOp);
              emptyTensor->setAttr("init", constOp.getValueAttr());
              fillOp.getResult(0).replaceAllUsesWith(emptyTensor.getResult());
            }
          }
        }
      }
    });

    for (unsigned i = 0; i < fillOpsToErase.size(); i++) {
      fillOpsToErase[i]->erase();
    }

    funcOp->walk([&](tensor::EmptyOp emptyOp) {
      if (emptyOp->hasOneUse()) {
        if (auto genericUser = dyn_cast<linalg::GenericOp>(
                emptyOp->getUses().begin()->getOwner())) {
          if (genericUser.getOutputs()[0] == emptyOp) {
            genericUser.getResult(0).replaceAllUsesWith(emptyOp);
          }
        }
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createSimplifyLinalgCopyPass() {
  return std::make_unique<SimplifyLinalgCopy>();
}