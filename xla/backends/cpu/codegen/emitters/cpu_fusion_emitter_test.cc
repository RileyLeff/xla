/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "llvm-c/Target.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/fusion_emitter.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/logical_buffer.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/types.h"

namespace xla {
namespace cpu {
namespace {

using ::testing::ElementsAreArray;

struct GetWorkDimensionsTestCase {
  std::string test_name;
  absl::string_view shape_and_layout;
  int64_t outer_dimension_partitions;
  uint64_t expected_num_work_groups;
  std::vector<int64_t> expected_work_tile_size;
  absl::string_view root_op = "copy(p0)";
};

class CpuFusionEmitterTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<GetWorkDimensionsTestCase> {
 protected:
  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& hlo) {
    return BufferAssigner::Run(
        &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
        [](const BufferValue& buffer) {
          return CpuExecutable::ShapeSizeBytes(buffer.shape());
        },
        &alias_info_, [](LogicalBuffer::Color) { return /*alignment=*/1; },
        BufferAssigner::Options{});
  }

  AliasInfo alias_info_;
};

static constexpr absl::string_view kScatterHlo = R"(
  add {
    %lhs = f32[] parameter(0)
    %rhs = f32[] parameter(1)
    ROOT %add.2 = f32[] add(%lhs, %rhs)
  }

  scatter_computation {
    %operand = f32[50,64,8] parameter(0)
    %indices = s32[500,1]{1,0} parameter(1)
    %updates = f32[500,1,64,8] parameter(2)
    ROOT %scatter = f32[50,64,8] scatter(%operand, %indices, %updates),
      update_window_dims={1,2,3},
      inserted_window_dims={},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      to_apply=add
  }

  ENTRY main {
    %p = f32[50,64,8]{2,1,0} parameter(0)
    %p.1 = s32[500,1]{1,0} parameter(1)
    %p.2 = f32[500,1,64,8]{3,2,1,0} parameter(2)
    ROOT %wrapped_scatter = f32[50,64,8]{2,1,0} fusion(%p, %p.1, %p.2),
      kind=kLoop,
      calls=%scatter_computation
  }
)";

TEST_F(CpuFusionEmitterTest, ScatterMlir) {
  constexpr absl::string_view kExpected = R"(
    CHECK:       module @wrapped_scatter attributes {{{.*}}xla.extra_backend_options = #xla<extra_backend_options["xla_cpu_disable_loop_unrolling"]>{{.*}}}
    CHECK:       @wrapped_scatter(
    CHECK-SAME:    xla.entry
    CHECK:           %[[XLA_LOOP:.+]] = xla.loop
    CHECK:           xla.pure_call
    CHECK:           scf.if
    CHECK:             xla.pure_call
    CHECK:             tensor.extract
    CHECK:             arith.addf
    CHECK:           return %[[XLA_LOOP]]
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kScatterHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
  TF_ASSERT_OK_AND_ASSIGN(KernelDefinition kernel_definition,
                          emitter.EmitKernelDefinition());
  const auto& mlir_source = kernel_definition.source();
  auto mlir_dump = mlir_source.ToString();
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(mlir_dump, kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(CpuFusionEmitterTest, ScatterLlvm) {
  constexpr absl::string_view kExpected = R"(
    CHECK-NOT:  @wrapped_scatter_entry(
    CHECK-NOT:  @wrapped_scatter_kernel(
    CHECK:      @wrapped_scatter(
    CHECK:      uwtable "frame-pointer"="all"
    CHECK-SAME: "prefer-vector-width"="512"
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kScatterHlo));
  auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_prefer_vector_width(512);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
  TF_ASSERT_OK_AND_ASSIGN(KernelDefinition kernel_definition,
                          emitter.EmitKernelDefinition());
  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  TF_ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  auto llvm_dump = llvm_source.ToString();
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(llvm_dump, kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(CpuFusionEmitterTest,
       ScatterReorderedOperandsWritesToResultSliceAndHonorsNonInvariant) {
  constexpr absl::string_view kHlo = R"(
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    scatter_computation {
      indices = s32[2,1] parameter(0)
      updates = s32[2,1] parameter(1)
      operand = s32[4] parameter(2)
      ROOT s = s32[4] scatter(operand, indices, updates),
        update_window_dims={1},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    }

    ENTRY main {
      p0 = s32[2,1] parameter(0)
      p1 = s32[2,1] parameter(1)
      p2 = s32[4] parameter(2)
      ROOT wrapped_scatter = s32[4] fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation,
        frontend_attributes={xla.no_invariant_operands="0"}
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> kernel_definition,
                       emitter.EmitKernelDefinition());
  EXPECT_FALSE(kernel_definition.spec().invariant_arguments().contains(0));
  EXPECT_TRUE(kernel_definition.spec().invariant_arguments().contains(1));
  EXPECT_FALSE(kernel_definition.spec().invariant_arguments().contains(2));
  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  const llvm::Function* wrapped =
      llvm_source.module()->getFunction("wrapped_scatter_wrapped");
  ASSERT_NE(wrapped, nullptr);
  // Operand 0 is marked not invariant, operand 1 is read only, operand 2 and
  // arg 3 share the writable result slice, and args 4..6 are the workgroup IDs.
  ASSERT_EQ(wrapped->arg_size(), 7);
  EXPECT_FALSE(wrapped->getArg(0)->onlyReadsMemory());
  EXPECT_FALSE(wrapped->getArg(0)->hasNoAliasAttr());
  EXPECT_TRUE(wrapped->getArg(1)->onlyReadsMemory());
  EXPECT_TRUE(wrapped->getArg(1)->hasNoAliasAttr());
  EXPECT_FALSE(wrapped->getArg(2)->onlyReadsMemory());
  EXPECT_TRUE(wrapped->getArg(2)->hasNoAliasAttr());
  EXPECT_FALSE(wrapped->getArg(2)->use_empty());
  EXPECT_TRUE(wrapped->getArg(3)->use_empty());
}

TEST_F(CpuFusionEmitterTest, SortComparatorMlirAndLlvm) {
  constexpr absl::string_view kSortHlo = R"(
    HloModule sort_module
    compare {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT lt = pred[] compare(lhs, rhs), direction=LT
    }
    ENTRY main {
      p0 = f32[8] parameter(0)
      ROOT sorted = f32[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK:      func.func @compare(
    CHECK-SAME:   %[[ARG0:.+]]: tensor<f32>, %[[ARG1:.+]]: tensor<f32>) -> i1
    CHECK-SAME:   xla.cpu.is_comparator
    CHECK-NOT:    xla.entry
    CHECK:        %[[LHS:.+]] = tensor.extract %[[ARG0]][] : tensor<f32>
    CHECK:        %[[RHS:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
    CHECK:        %[[CMP:.+]] = xla.pure_call @compare_lt(%[[LHS]], %[[RHS]]) : (f32, f32) -> i8
    CHECK:        %[[RES:.+]] = arith.trunci %[[CMP]] : i8 to i1
    CHECK:        return %[[RES]] : i1
  )";
  constexpr absl::string_view kExpectedLlvm = R"(
    CHECK:      define zeroext i1 @compare(ptr noundef nonnull %{{.+}}) #[[ATTR:[0-9]+]]
    CHECK:      attributes #[[ATTR]] = {{{.*}}uwtable{{[ }]}}
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kSortHlo));
  const HloComputation* comparator =
      hlo_module->entry_computation()->root_instruction()->to_apply();
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource mlir_source,
      EmitSortComparator(*mlir_context, *comparator, "compare"));
  EXPECT_THAT(RunFileCheck(mlir_source.ToString(), kExpectedMlir),
              absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(mlir_source)));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), kExpectedLlvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, SortComparatorWeakAndTotalOrder) {
  constexpr absl::string_view kWeakSortHlo = R"hlo(
    HloModule sort_weak_module
    compare {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT lt = pred[] compare(lhs, rhs), direction=LT, order=WEAK
    }
    ENTRY main {
      p0 = bf16[8] parameter(0)
      ROOT sorted = bf16[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )hlo";
  constexpr absl::string_view kExpectedWeakMlir = R"(
    CHECK:      func.func private @compare_lt(
    CHECK-DAG:    arith.cmpf ult
    CHECK-DAG:    arith.cmpf ord
    CHECK:        arith.andi
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> weak_module,
                       ParseAndReturnVerifiedModule(kWeakSortHlo));
  const HloComputation* weak_comparator =
      weak_module->entry_computation()->root_instruction()->to_apply();
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource weak_mlir,
      EmitSortComparator(*mlir_context, *weak_comparator, "compare_weak"));
  EXPECT_THAT(RunFileCheck(weak_mlir.ToString(), kExpectedWeakMlir),
              absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource weak_llvm,
                       compiler.Compile(std::move(weak_mlir)));
  EXPECT_THAT(RunFileCheck(weak_llvm.ToString(),
                           "CHECK: define zeroext i1 @compare_weak("),
              absl_testing::IsOkAndHolds(true));

  constexpr absl::string_view kTotalSortHlo = R"hlo(
    HloModule sort_total_module
    compare {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT lt = pred[] compare(lhs, rhs), direction=LT, order=TOTAL
    }
    ENTRY main {
      p0 = bf16[8] parameter(0)
      ROOT sorted = bf16[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )hlo";
  constexpr absl::string_view kExpectedTotalMlir = R"(
    CHECK:      func.func private @compare_lt(
    CHECK:        arith.bitcast %{{.+}} : bf16 to i16
    CHECK:        arith.cmpi slt
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> total_module,
                       ParseAndReturnVerifiedModule(kTotalSortHlo));
  const HloComputation* total_comparator =
      total_module->entry_computation()->root_instruction()->to_apply();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource total_mlir,
      EmitSortComparator(*mlir_context, *total_comparator, "compare_total"));
  EXPECT_THAT(RunFileCheck(total_mlir.ToString(), kExpectedTotalMlir),
              absl_testing::IsOkAndHolds(true));
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource total_llvm,
                       compiler.Compile(std::move(total_mlir)));
  EXPECT_THAT(RunFileCheck(total_llvm.ToString(),
                           "CHECK: define zeroext i1 @compare_total("),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, SortComparatorUnsupportedOp) {
  constexpr absl::string_view kSortHlo = R"(
    HloModule sort_module
    cond {
      state = s32[] parameter(0)
      limit = s32[] constant(4)
      ROOT lt = pred[] compare(state, limit), direction=LT
    }
    body {
      state = s32[] parameter(0)
      one = s32[] constant(1)
      ROOT next = s32[] add(state, one)
    }
    compare {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      loop = s32[] while(lhs), condition=cond, body=body
      ROOT lt = pred[] compare(loop, rhs), direction=LT
    }
    ENTRY main {
      p0 = s32[8] parameter(0)
      ROOT sorted = s32[8] sort(p0), dimensions={0}, to_apply=compare
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kSortHlo));
  const HloComputation* comparator =
      hlo_module->entry_computation()->root_instruction()->to_apply();
  auto mlir_context = FusionCompiler::CreateContext();
  EXPECT_THAT(EmitSortComparator(*mlir_context, *comparator, "compare"),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_P(CpuFusionEmitterTest, WorkDimensionsTest) {
  const GetWorkDimensionsTestCase& tc = GetParam();
  constexpr absl::string_view kHloTemplate = R"(
    HloModule test
    fused_computation {
      p0 = $0 parameter(0)
      ROOT res = $0 $2
    }
    ENTRY main {
      p0 = $0 parameter(0)
      ROOT fusion = $0 fusion(p0), kind=kLoop,
        calls=fused_computation,
        backend_config={"outer_dimension_partitions":["$1"]}
    }
  )";
  std::string hlo = absl::Substitute(kHloTemplate, tc.shape_and_layout,
                                     tc.outer_dimension_partitions, tc.root_op);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*module));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  const WorkDimensions& dims = kernel_definition.spec().work_dimensions();
  EXPECT_EQ(dims.num_work_groups.x, tc.expected_num_work_groups);
  EXPECT_THAT(dims.work_tile_size.dimensions,
              ElementsAreArray(tc.expected_work_tile_size));
}

INSTANTIATE_TEST_SUITE_P(
    WorkDimensions, CpuFusionEmitterTest,
    ::testing::ValuesIn<GetWorkDimensionsTestCase>({
        {
            /*test_name=*/"NonSubByteDoesNotRoundOddTile",
            /*shape_and_layout=*/"f32[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{3, 3},
        },
        {
            // CeilOfRatio(6, 2) = 3 is rounded up to 4 (multiple of 2
            // elements/byte), giving 4 * 3 = 12 elements (6 whole bytes) per
            // workgroup.
            /*test_name=*/"SubByteS4RoundsOddSplitTileToMultipleOfTwo",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // CeilOfRatio(5, 2) = 3 is also rounded up to 4, so workgroup 0
            // writes 12 elements (6 whole bytes) and workgroup 1 writes the
            // remaining 3 elements starting at byte 6.
            /*test_name=*/
            "SubByteS4OddRowsStillRoundsSplitTileToMultipleOfTwo",
            /*shape_and_layout=*/"s4[5,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // CeilOfRatio(6, 2) = 3 is rounded up to 4 (multiple of 4
            // elements/byte), giving 4 * 3 = 12 elements (3 whole bytes) per
            // workgroup.
            /*test_name=*/"SubByteS2RoundsSplitTileToMultipleOfFour",
            /*shape_and_layout=*/"s2[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // CeilOfRatio(12, 2) = 6 is rounded up to 8 (multiple of 8
            // elements/byte), giving 8 * 3 = 24 elements (3 whole bytes) per
            // workgroup.
            /*test_name=*/"SubByteS1RoundsSplitTileToMultipleOfEight",
            /*shape_and_layout=*/"s1[12,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{8, 3},
        },
        {
            // PRED is stored 1 element per byte, so odd split tiles are
            // not rounded.
            /*test_name=*/"PredIsUnpackedAndDoesNotRoundOddTile",
            /*shape_and_layout=*/"pred[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{3, 3},
        },
        {
            // CeilOfRatio(6, 4) = 2 (already a multiple of 2 elems/byte);
            // 3 workgroups of size 2 cover all 6 rows, so num_work_groups is
            // shrunk from 4 to 3.
            /*test_name=*/"SubByteRecomputesNumWorkGroupsAfterRounding",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/4,
            /*expected_num_work_groups=*/3,
            /*expected_work_tile_size=*/{2, 3},
        },
        {
            // Concatenate fusions with packed sub-byte outputs must run in a
            // single workgroup covering the entire shape, because concatenated
            // operands can start at non-byte-aligned offsets.
            /*test_name=*/"ConcatenateSubByteForcesSingleWorkGroup",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{6, 3},
            /*root_op=*/"concatenate(p0), dimensions={0}",
        },
        {
            // DynamicUpdateSlice fusions with packed sub-byte outputs must also
            // run in a single workgroup covering the entire shape, because the
            // update index may land at a non-byte aligned offset.
            /*test_name=*/"DynamicUpdateSliceSubByteForcesSingleWorkGroup",
            /*shape_and_layout=*/"s4[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{6, 3},
            /*root_op=*/
            "dynamic-update-slice(p0, p0, s32[] constant(0), s32[] "
            "constant(0))",
        },
        {
            // 3D shape where the leading dim (2) is smaller than 4 workgroups:
            // dim 0 (2) is folded with dim 1 (6) into 12 slices of 3 elements;
            // CeilOfRatio(12, 4) = 3 rounds up to 4, leaving 3 workgroups of
            // tile {4, 3}.
            /*test_name=*/"ThreeDimsFoldsLeadingDimAndRoundsSplitTile",
            /*shape_and_layout=*/"s4[2,6,3]{2,1,0}",
            /*outer_dimension_partitions=*/4,
            /*expected_num_work_groups=*/3,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // 3D shape where the leading dim (6) is large enough to split
            // directly: CeilOfRatio(6, 2) = 3 rounds up to 4, keeping both
            // minor dims (4 and 3) intact.
            /*test_name=*/"ThreeDimsSplitsOutermostDimAndKeepsMinorDims",
            /*shape_and_layout=*/"s4[6,4,3]{2,1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 4, 3},
        },
        {
            // More workgroups (8) than the product of the first two dimensions
            // (2 * 2 = 4): all three dimensions fold together (2 * 2 * 3 = 12),
            // and CeilOfRatio(12, 8) = 2 yields 6 workgroups of 1D tile {2}.
            /*test_name=*/"FoldsAllDimensionsIntoSingleTileDim",
            /*shape_and_layout=*/"s4[2,2,3]{2,1,0}",
            /*outer_dimension_partitions=*/8,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{2},
        },
        {
            // As many workgroups (12) as total elements (2 * 2 * 3 = 12): all
            // three dimensions fold together, and CeilOfRatio(12, 12) = 1 is
            // rounded up to 2 elements (1 byte) per workgroup, shrinking
            // num_work_groups from 12 to 6.
            /*test_name=*/"OneWorkGroupPerElementRoundsToWholeByte",
            /*shape_and_layout=*/"s4[2,2,3]{2,1,0}",
            /*outer_dimension_partitions=*/12,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{2},
        },
        {
            // Column-major layout {0,1}: dim 1 (size 6) is the most-major
            // physical dimension and gets split and rounded from 3 to 4, while
            // dim 0 (size 3) is minor.
            /*test_name=*/"SubByteColumnMajorLayoutSplitsPhysicalMajorDim",
            /*shape_and_layout=*/"s4[3,6]{0,1}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{4, 3},
        },
        {
            // Single workgroup does not round odd dimensions.
            /*test_name=*/"SingleWorkGroupDoesNotRoundOddSubByteDims",
            /*shape_and_layout=*/"s4[5,3]{1,0}",
            /*outer_dimension_partitions=*/1,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{5, 3},
        },
        {
            // f6 types are stored one element per byte and are not rounded.
            /*test_name=*/"NonPowerOfTwoSubByteF6DoesNotRoundOddTile",
            /*shape_and_layout=*/"f6e3m2fn[6,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/2,
            /*expected_work_tile_size=*/{3, 3},
        },
        {
            // More workgroups (16) than elements (12): clamped to 12, then
            // rounded to 2 elements (1 byte) per workgroup.
            /*test_name=*/"MoreWorkGroupsThanElementsSubByteRoundsToWholeByte",
            /*shape_and_layout=*/"s4[2,2,3]{2,1,0}",
            /*outer_dimension_partitions=*/16,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{2},
        },
        {
            // More workgroups (8) than elements (6): clamped to 6.
            /*test_name=*/
            "MoreWorkGroupsThanElementsNonSubByteClampsToElements",
            /*shape_and_layout=*/"f32[2,3]{1,0}",
            /*outer_dimension_partitions=*/8,
            /*expected_num_work_groups=*/6,
            /*expected_work_tile_size=*/{1},
        },
        {
            // Zero-element shapes get a single workgroup and an empty tile.
            /*test_name=*/"ZeroElementsUsesSingleWorkGroup",
            /*shape_and_layout=*/"f32[0,3]{1,0}",
            /*outer_dimension_partitions=*/2,
            /*expected_num_work_groups=*/1,
            /*expected_work_tile_size=*/{},
        },
    }),
    [](const ::testing::TestParamInfo<GetWorkDimensionsTestCase>& info) {
      return info.param.test_name;
    });

TEST_F(CpuFusionEmitterTest, ConcatenateMatchingLayoutEmitsMemcpy) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[4,2,8]{2,1,0} parameter(0)
      p1 = f32[4,3,8]{2,1,0} parameter(1)
      ROOT concat = f32[4,5,8]{2,1,0} concatenate(p0, p1), dimensions={1}
    }
    ENTRY main {
      p0 = f32[4,2,8]{2,1,0} parameter(0)
      p1 = f32[4,3,8]{2,1,0} parameter(1)
      ROOT wrapped_concat = f32[4,5,8]{2,1,0} fusion(p0, p1),
        kind=kLoop, calls=fused_computation,
        backend_config={"outer_dimension_partitions":["2"]}
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK:      func.func @wrapped_concat(
    CHECK-SAME:   xla.entry
    CHECK-NOT:    xla.loop
    CHECK:        scf.for
    CHECK:          xla.copy_slice {{.*}} count(16)
    CHECK:          xla.copy_slice {{.*}} count(24)
  )";
  std::string expected_llvm = absl::Substitute(R"(
    CHECK:      define ptr @wrapped_concat(
    CHECK-NOT:    call void @llvm.memcpy
    CHECK:        call void @llvm.assume(i1 true) [ "align"(ptr %{{.*}}, i64 $0) ]
    CHECK:      define internal void @wrapped_concat_wrapped(
    CHECK:        call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 64, i1 false)
    CHECK:        call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 96, i1 false)
  )",
                                               MinAlign());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_EQ(kernel_definition.spec().work_dimensions().num_work_groups.x, 2);
  EXPECT_THAT(
      RunFileCheck(kernel_definition.source().ToString(), kExpectedMlir),
      absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), expected_llvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest,
       ConcatenateNoOuterDimsForcesSingleWorkGroupAndEmitsMemcpy) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[8,16]{1,0} parameter(0)
      p1 = f32[12,16]{1,0} parameter(1)
      ROOT concat = f32[20,16]{1,0} concatenate(p0, p1), dimensions={0}
    }
    ENTRY main {
      p0 = f32[8,16]{1,0} parameter(0)
      p1 = f32[12,16]{1,0} parameter(1)
      ROOT wrapped_concat = f32[20,16]{1,0} fusion(p0, p1),
        kind=kLoop, calls=fused_computation,
        backend_config={"outer_dimension_partitions":["4"]}
    }
  )";
  std::string expected_llvm = absl::Substitute(R"(
    CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align $0 %{{.*}}, ptr align $0 %{{.*}}, i64 512, i1 false)
    CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align $0 %{{.*}}, ptr align $0 %{{.*}}, i64 768, i1 false)
  )",
                                               MinAlign());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_EQ(kernel_definition.spec().work_dimensions().num_work_groups.x, 1);

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), expected_llvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ConcatenateMismatchedLayoutFallsBackToLoop) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[4,8]{0,1} parameter(0)
      p1 = f32[6,8]{1,0} parameter(1)
      ROOT concat = f32[10,8]{1,0} concatenate(p0, p1), dimensions={0}
    }
    ENTRY main {
      p0 = f32[4,8]{0,1} parameter(0)
      p1 = f32[6,8]{1,0} parameter(1)
      ROOT wrapped_concat = f32[10,8]{1,0} fusion(p0, p1),
        kind=kLoop, calls=fused_computation
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK-NOT: xla.copy_slice
    CHECK:     xla.loop
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_THAT(
      RunFileCheck(kernel_definition.source().ToString(), kExpectedMlir),
      absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest,
       ConcatenateNonDefaultMatchingLayoutUsesPhysicalOrderMemcpy) {
  constexpr absl::string_view kHlo = R"(
    HloModule concat_module
    fused_computation {
      p0 = f32[2,4,3]{0,2,1} parameter(0)
      p1 = f32[2,4,5]{0,2,1} parameter(1)
      ROOT concat = f32[2,4,8]{0,2,1} concatenate(p0, p1), dimensions={2}
    }
    ENTRY main {
      p0 = f32[2,4,3]{0,2,1} parameter(0)
      p1 = f32[2,4,5]{0,2,1} parameter(1)
      ROOT wrapped_concat = f32[2,4,8]{0,2,1} fusion(p0, p1),
        kind=kLoop, calls=fused_computation,
        backend_config={"outer_dimension_partitions":["2"]}
    }
  )";
  constexpr absl::string_view kExpectedMlir = R"(
    CHECK:      func.func @wrapped_concat(
    CHECK-NOT:    xla.loop
    CHECK:        scf.for
    CHECK:          xla.copy_slice {{.*}} count(6)
    CHECK:          xla.copy_slice {{.*}} count(10)
  )";
  constexpr absl::string_view kExpectedLlvm = R"(
    CHECK: call void @llvm.memcpy{{.*}}i64 24
    CHECK: call void @llvm.memcpy{{.*}}i64 40
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  const auto* fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> kernel_definition,
      EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  EXPECT_EQ(kernel_definition.spec().work_dimensions().num_work_groups.x, 2);
  EXPECT_THAT(
      RunFileCheck(kernel_definition.source().ToString(), kExpectedMlir),
      absl_testing::IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource llvm_source,
      compiler.Compile(std::move(kernel_definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), kExpectedLlvm),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(CpuFusionEmitterTest, ConcatenateComplexAndNarrowFloatCompileToMemcpy) {
  for (absl::string_view elem_type : {"c64", "f6e3m2fn"}) {
    std::string hlo = absl::Substitute(R"(
      HloModule concat_module
      fused_computation {
        p0 = $0[4,3]{1,0} parameter(0)
        p1 = $0[4,5]{1,0} parameter(1)
        ROOT concat = $0[4,8]{1,0} concatenate(p0, p1), dimensions={1}
      }
      ENTRY main {
        p0 = $0[4,3]{1,0} parameter(0)
        p1 = $0[4,5]{1,0} parameter(1)
        ROOT wrapped_concat = $0[4,8]{1,0} fusion(p0, p1),
          kind=kLoop, calls=fused_computation
      }
    )",
                                       elem_type);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                         RunBufferAssignment(*hlo_module));
    const auto* fusion = Cast<HloFusionInstruction>(
        hlo_module->entry_computation()->root_instruction());
    auto mlir_context = FusionCompiler::CreateContext();
    ASSERT_OK_AND_ASSIGN(
        KernelDefinition<MlirKernelSource> kernel_definition,
        EmitFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                         /*use_unique_c_name=*/false,
                         /*enable_tiled_emitter=*/false));
    FusionCompiler compiler(mlir_context.get(),
                            FusionCompiler::Options{512, 1, true});
    ASSERT_OK_AND_ASSIGN(
        LlvmKernelSource llvm_source,
        compiler.Compile(std::move(kernel_definition).TakeSource()));
    EXPECT_THAT(
        RunFileCheck(llvm_source.ToString(),
                     elem_type == "c64"
                         ? "CHECK: call void @llvm.memcpy{{.*}}i64 24\nCHECK: "
                           "call void @llvm.memcpy{{.*}}i64 40"
                         : "CHECK: call void @llvm.memcpy{{.*}}i64 3\nCHECK: "
                           "call void @llvm.memcpy{{.*}}i64 5"),
        absl_testing::IsOkAndHolds(true));
  }
}

ABSL_ATTRIBUTE_NO_SANITIZE_CFI XLA_CPU_KernelError* CallCpuKernel(
    FunctionLibrary::Kernel* kernel, XLA_CPU_KernelCallFrame* frame) {
  return (*kernel)(frame);
}

TEST_F(CpuFusionEmitterTest,
       AtomicCasEmitsAlignedUnorderedLoadsAndJitExecutes) {
  constexpr absl::string_view kMlir = R"(
    module attributes {xla.cpu_memory_region_name = "cpu_fusion_emitter_test"} {
      func.func @cpu_atomic_cas(
          %s4_buf: tensor<4xi4> {llvm.align = 16 : index, xla.slice_index = 0},
          %f16_buf: tensor<2xf16> {llvm.align = 16 : index, xla.slice_index = 1},
          %f32_buf: tensor<2xf32> {llvm.align = 16 : index, xla.slice_index = 2},
          %c64_buf: tensor<2xcomplex<f32>> {llvm.align = 16 : index, xla.slice_index = 3})
          -> (tensor<4xi4>, tensor<2xf16>, tensor<2xf32>,
              tensor<2xcomplex<f32>>)
          attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %v_i4 = arith.constant 5 : i4
        %v_f16 = arith.constant 2.5 : f16
        %v_f32 = arith.constant 3.5 : f32
        %v_c64 = complex.constant [4.0 : f32, 6.0 : f32] : complex<f32>

        %out_s4 = xla.atomic_store %v_i4 into %s4_buf[%c1] : tensor<4xi4>
        %out_f16 = xla.atomic_rmw %f16_buf[%c1] : tensor<2xf16> {
          ^bb0(%cur: f16):
            %sum = arith.addf %cur, %v_f16 : f16
            xla.yield %sum : f16
        }
        %out_f32 = xla.atomic_rmw %f32_buf[%c0] : tensor<2xf32> {
          ^bb0(%cur: f32):
            %max = arith.maximumf %cur, %v_f32 : f32
            xla.yield %max : f32
        }
        %out_c64 = xla.atomic_rmw %c64_buf[%c1] : tensor<2xcomplex<f32>> {
          ^bb0(%cur: complex<f32>):
            %sum = complex.add %cur, %v_c64 : complex<f32>
            xla.yield %sum : complex<f32>
        }
        return %out_s4, %out_f16, %out_f32, %out_c64
            : tensor<4xi4>, tensor<2xf16>, tensor<2xf32>,
              tensor<2xcomplex<f32>>
      }
    }
  )";

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  mlir::MLIRContext* context_ptr = mlir_context.get();
  ASSERT_OK_AND_ASSIGN(
      MlirKernelSource mlir_source,
      MlirKernelSource::ParseFromString(kMlir, std::move(mlir_context)));
  FusionCompiler compiler(context_ptr, FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(mlir_source)));

  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), R"(
    CHECK: load atomic i8, ptr %[[S4_ADDR:.*]] unordered, align 1
    CHECK: cmpxchg ptr %[[S4_ADDR]], i8 %{{.*}}, i8 %{{.*}} monotonic monotonic
    CHECK: load atomic i16, ptr %[[F16_ADDR:.*]] unordered, align 2
    CHECK: cmpxchg ptr %[[F16_ADDR]], i16 %{{.*}}, i16 %{{.*}} monotonic monotonic
    CHECK: load atomic i32, ptr %[[F32_ADDR:.*]] unordered, align 4
    CHECK: cmpxchg ptr %[[F32_ADDR]], i32 %{{.*}}, i32 %{{.*}} monotonic monotonic
    CHECK: load atomic i64, ptr %[[C64_ADDR:.*]] unordered, align 8
    CHECK: cmpxchg ptr %[[C64_ADDR]], i64 %{{.*}}, i64 %{{.*}} monotonic monotonic
  )"),
              absl_testing::IsOkAndHolds(true));

  std::unique_ptr<IrCompiler> ir_compiler =
      IrCompiler::Create({},
                         IrCompiler::Options{
                             /*opt_level=*/llvm::CodeGenOptLevel::Default,
                             /*optimize_for_size=*/false,
                             TargetMachineOptions(GetDebugOptionsFromFlags()),
                         },
                         IrCompiler::CompilationHooks());
  ASSERT_OK_AND_ASSIGN(JitCompiler jit,
                       JitCompiler::Create({}, std::move(ir_compiler)));
  ASSERT_OK(jit.AddModule(std::move(llvm_source).thread_safe_module()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionLibrary> library,
      std::move(jit).Compile(
          {FunctionLibrary::Sym<FunctionLibrary::Kernel>("cpu_atomic_cas")}));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("cpu_atomic_cas"));

  alignas(16) std::array<uint8_t, 2> s4_bytes = {0xa3, 0x72};
  alignas(16) std::array<half, 2> f16_vals = {half(1.0f), half(1.5f)};
  alignas(16) std::array<float, 2> f32_vals = {1.25f, 9.0f};
  alignas(16) std::array<complex64, 2> c64_vals = {complex64(1.0f, 2.0f),
                                                   complex64(3.0f, 5.0f)};
  std::vector<XLA_CPU_KernelArg> args = {
      {s4_bytes.data(), s4_bytes.size()},
      {f16_vals.data(), f16_vals.size() * sizeof(half)},
      {f32_vals.data(), f32_vals.size() * sizeof(float)},
      {c64_vals.data(), c64_vals.size() * sizeof(complex64)},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ(CallCpuKernel(kernel, &frame), nullptr);

  EXPECT_EQ(s4_bytes[0], 0x53);
  EXPECT_EQ(s4_bytes[1], 0x72);
  EXPECT_EQ(f16_vals[0], half(1.0f));
  EXPECT_EQ(f16_vals[1], half(4.0f));
  EXPECT_EQ(f32_vals[0], 3.5f);
  EXPECT_EQ(f32_vals[1], 9.0f);
  EXPECT_EQ(c64_vals[0], complex64(1.0f, 2.0f));
  EXPECT_EQ(c64_vals[1], complex64(7.0f, 11.0f));
}

void InitializeCrossCpuTargets() {
  static absl::once_flag init_once;
  absl::call_once(init_once, []() {
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64TargetMC();
    LLVMInitializeAArch64AsmPrinter();
  });
}

constexpr absl::string_view kCpuAtomic128StoreMlir = R"(
  module attributes {xla.cpu_memory_region_name = "cpu_atomic_128_store"} {
    func.func @cpu_atomic_128_store(
        %i128_buf: tensor<4xi128> {llvm.align = 16 : index, xla.slice_index = 0},
        %i128_src: tensor<1xi128> {llvm.align = 16 : index, xla.invariant, xla.slice_index = 1},
        %c128_buf: tensor<4xcomplex<f64>> {llvm.align = 16 : index, xla.slice_index = 2},
        %c128_src: tensor<1xcomplex<f64>> {llvm.align = 16 : index, xla.invariant, xla.slice_index = 3})
        -> (tensor<4xi128>, tensor<4xcomplex<f64>>)
        attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %v_i128 = tensor.extract %i128_src[%c0] : tensor<1xi128>
      %v_c128 = tensor.extract %c128_src[%c0] : tensor<1xcomplex<f64>>
      %out_i128 = xla.atomic_store %v_i128 into %i128_buf[%c1] : tensor<4xi128>
      %out_c128 = xla.atomic_store %v_c128 into %c128_buf[%c2] : tensor<4xcomplex<f64>>
      return %out_i128, %out_c128 : tensor<4xi128>, tensor<4xcomplex<f64>>
    }
  }
)";

constexpr absl::string_view kCpuAtomic128RmwMlir = R"(
  module attributes {xla.cpu_memory_region_name = "cpu_atomic_128_rmw"} {
    func.func @cpu_atomic_128_rmw(
        %i128_buf: tensor<4xi128> {llvm.align = 16 : index, xla.slice_index = 0},
        %i128_delta: tensor<1xi128> {llvm.align = 16 : index, xla.invariant, xla.slice_index = 1},
        %c128_buf: tensor<4xcomplex<f64>> {llvm.align = 16 : index, xla.slice_index = 2},
        %c128_delta: tensor<1xcomplex<f64>> {llvm.align = 16 : index, xla.invariant, xla.slice_index = 3})
        -> (tensor<4xi128>, tensor<4xcomplex<f64>>)
        attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %d_i128 = tensor.extract %i128_delta[%c0] : tensor<1xi128>
      %d_c128 = tensor.extract %c128_delta[%c0] : tensor<1xcomplex<f64>>
      %out_i128 = xla.atomic_rmw %i128_buf[%c1] : tensor<4xi128> {
        ^bb0(%cur: i128):
          %sum = arith.addi %cur, %d_i128 : i128
          xla.yield %sum : i128
      }
      %out_c128 = xla.atomic_rmw %c128_buf[%c2] : tensor<4xcomplex<f64>> {
        ^bb0(%cur: complex<f64>):
          %sum = complex.add %cur, %d_c128 : complex<f64>
          xla.yield %sum : complex<f64>
      }
      return %out_i128, %out_c128 : tensor<4xi128>, tensor<4xcomplex<f64>>
    }
  }
)";

struct CompiledAtomic128Artifacts {
  std::string llvm_ir;
  std::string assembly;
  std::vector<std::string> undefined_symbols;
  std::vector<std::string> helper_symbols;
};

absl::StatusOr<std::unique_ptr<llvm::TargetMachine>> CreateCrossTargetMachine(
    absl::string_view triple_str, absl::string_view cpu,
    absl::string_view features, llvm::CodeGenOptLevel opt_level) {
  InitializeCrossCpuTargets();
  llvm::Triple triple(llvm::StringRef(triple_str.data(), triple_str.size()));
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (target == nullptr) {
    return absl::InternalError(absl::StrCat("lookupTarget failed: ", error));
  }
  llvm::TargetOptions target_options;
  std::unique_ptr<llvm::TargetMachine> tm =
      absl::WrapUnique(target->createTargetMachine(
          triple, cpu, features, target_options, llvm::Reloc::PIC_,
          /*CM=*/std::nullopt, opt_level));
  TF_RET_CHECK(tm != nullptr) << "createTargetMachine returned null";
  return std::move(tm);
}

absl::StatusOr<LlvmKernelSource> LowerAtomic128ForTarget(
    absl::string_view mlir_text, const llvm::TargetMachine& tm) {
  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  ABSL_ASSIGN_OR_RETURN(
      MlirKernelSource mlir_source,
      MlirKernelSource::ParseFromString(mlir_text, std::move(mlir_context)));

  FusionCompiler::Options options;
  options.vector_width = 256;
  options.verification_level = 1;
  options.fast_min_max = true;
  options.cpu_features = tm.getTargetFeatureString().str();
  options.cpu_device_spec = CpuDeviceSpecFromTargetMachine(&tm);

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  FusionCompiler compiler(mlir_source.module().getContext(),
                          std::move(options));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<llvm::Module> llvm_module,
                        compiler.Compile(*llvm_context, mlir_source.module()));
  return LlvmKernelSource(std::move(llvm_context), std::move(llvm_module));
}

absl::StatusOr<CompiledAtomic128Artifacts> CompileAtomic128ForTarget(
    absl::string_view mlir_text, absl::string_view triple_str,
    absl::string_view cpu, absl::string_view features,
    llvm::CodeGenOptLevel opt_level = llvm::CodeGenOptLevel::Default) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<llvm::TargetMachine> tm,
      CreateCrossTargetMachine(triple_str, cpu, features, opt_level));
  ABSL_ASSIGN_OR_RETURN(LlvmKernelSource llvm_source,
                        LowerAtomic128ForTarget(mlir_text, *tm));

  CompiledAtomic128Artifacts out;
  out.llvm_ir = llvm_source.ToString();

  std::unique_ptr<llvm::Module> optimized_clone;
  IrCompiler::CompilationHooks hooks;
  hooks.post_optimization = [&](const llvm::Module& m) {
    optimized_clone = llvm::CloneModule(m);
  };
  std::string triple_copy(triple_str);
  std::string cpu_copy(cpu);
  std::string features_copy(features);
  IrCompiler ir_compiler(
      [triple_copy, cpu_copy, features_copy, opt_level]() {
        return CreateCrossTargetMachine(triple_copy, cpu_copy, features_copy,
                                        opt_level);
      },
      IrCompiler::Options{
          opt_level,
          /*optimize_for_size=*/false,
          TargetMachineOptions(triple_str, cpu, features),
      },
      std::move(hooks));

  llvm::orc::ThreadSafeModule tsm = std::move(llvm_source).thread_safe_module();
  llvm::Module* module = tsm.getModuleUnlocked();
  module->setTargetTriple(tm->getTargetTriple());
  module->setDataLayout(tm->createDataLayout());

  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> obj_buf =
      ir_compiler(*module);
  if (!obj_buf) {
    return absl::InternalError(llvm::toString(obj_buf.takeError()));
  }
  llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> obj_file =
      llvm::object::ObjectFile::createObjectFile((*obj_buf)->getMemBufferRef());
  if (!obj_file) {
    return absl::InternalError(llvm::toString(obj_file.takeError()));
  }
  for (const llvm::object::SymbolRef& sym : (*obj_file)->symbols()) {
    llvm::Expected<uint32_t> flags = sym.getFlags();
    if (!flags) {
      return absl::InternalError(llvm::toString(flags.takeError()));
    }
    llvm::Expected<llvm::StringRef> name = sym.getName();
    if (!name) {
      return absl::InternalError(llvm::toString(name.takeError()));
    }
    if (name->empty()) {
      continue;
    }
    if ((*flags & llvm::object::SymbolRef::SF_Undefined) != 0) {
      out.undefined_symbols.push_back(name->str());
    }
    if (name->contains("__atomic_") || name->contains("__aarch64_") ||
        name->contains("__sync_")) {
      out.helper_symbols.push_back(name->str());
    }
  }

  TF_RET_CHECK(optimized_clone != nullptr);
  llvm::SmallVector<char, 0> asm_chars;
  llvm::raw_svector_ostream asm_stream(asm_chars);
  llvm::legacy::PassManager asm_pm;
  TF_RET_CHECK(!tm->addPassesToEmitFile(asm_pm, asm_stream, nullptr,
                                        llvm::CodeGenFileType::AssemblyFile));
  asm_pm.run(*optimized_clone);
  optimized_clone.reset();
  out.assembly.assign(asm_chars.data(), asm_chars.size());
  return out;
}

void ExpectNoAtomicRuntimeHelpers(const CompiledAtomic128Artifacts& artifacts) {
  EXPECT_THAT(artifacts.undefined_symbols, ::testing::IsEmpty())
      << artifacts.assembly;
  EXPECT_THAT(artifacts.helper_symbols, ::testing::IsEmpty())
      << artifacts.assembly;
  EXPECT_THAT(artifacts.assembly,
              ::testing::Not(::testing::HasSubstr("__atomic_")));
  EXPECT_THAT(artifacts.assembly,
              ::testing::Not(::testing::HasSubstr("__aarch64_")));
  EXPECT_THAT(artifacts.assembly,
              ::testing::Not(::testing::HasSubstr("__sync_")));
}

TEST_F(CpuFusionEmitterTest, Atomic128X86_64Codegen) {
  constexpr absl::string_view kTriple = "x86_64-unknown-linux-gnu";

  // x86_64 with +cx16 (without +avx): both store and RMW lower via 16 byte
  // aligned CAS loops to native lock cmpxchg16b with zero libcalls.
  for (llvm::CodeGenOptLevel opt :
       {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
    ASSERT_OK_AND_ASSIGN(
        CompiledAtomic128Artifacts store_cx16,
        CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, "x86-64",
                                  "+cx16", opt));
    ExpectNoAtomicRuntimeHelpers(store_cx16);
    EXPECT_THAT(RunFileCheck(store_cx16.llvm_ir, R"(
      CHECK: load atomic i128, ptr %[[I_ADDR:.*]] unordered, align 16
      CHECK: cmpxchg ptr %[[I_ADDR]], i128 %{{.*}}, i128 %{{.*}} monotonic monotonic, align 16
      CHECK: load atomic i128, ptr %[[C_ADDR:.*]] unordered, align 16
      CHECK: cmpxchg ptr %[[C_ADDR]], i128 %{{.*}}, i128 %{{.*}} monotonic monotonic, align 16
    )"),
                absl_testing::IsOkAndHolds(true));
    EXPECT_THAT(store_cx16.assembly,
                ::testing::ContainsRegex("lock[[:space:]]+cmpxchg16b"));
    EXPECT_THAT(store_cx16.assembly,
                ::testing::Not(::testing::HasSubstr("vmovdqa")));

    ASSERT_OK_AND_ASSIGN(
        CompiledAtomic128Artifacts rmw_cx16,
        CompileAtomic128ForTarget(kCpuAtomic128RmwMlir, kTriple, "x86-64",
                                  "+cx16", opt));
    ExpectNoAtomicRuntimeHelpers(rmw_cx16);
    EXPECT_THAT(rmw_cx16.assembly,
                ::testing::ContainsRegex("lock[[:space:]]+cmpxchg16b"));
  }

  // x86_64 with +cx16,+avx (explicit and implied by haswell and
  // skylake-avx512): store lowers to direct vmovdqa (no cmpxchg16b), and RMW
  // uses vmovdqa for the initial load plus lock cmpxchg16b for the CAS update.
  for (const auto& [cpu, features] :
       std::vector<std::pair<absl::string_view, absl::string_view>>{
           {"x86-64", "+cx16,+avx"}, {"haswell", ""}, {"skylake-avx512", ""}}) {
    for (llvm::CodeGenOptLevel opt :
         {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts store_avx,
          CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, cpu,
                                    features, opt));
      ExpectNoAtomicRuntimeHelpers(store_avx);
      EXPECT_THAT(RunFileCheck(store_avx.llvm_ir, R"(
        CHECK-NOT: cmpxchg
        CHECK: store atomic i128 %{{.*}}, ptr %{{.*}} unordered, align 16
        CHECK-NOT: cmpxchg
        CHECK: store atomic i128 %{{.*}}, ptr %{{.*}} unordered, align 16
        CHECK-NOT: cmpxchg
      )"),
                  absl_testing::IsOkAndHolds(true));
      // SelectionDAG emits vmovdqa for 16 byte aligned 128 bit atomic stores;
      // at O2 X86ExecutionDomainFix may replace a pure copy store with the
      // equivalent aligned vmovaps/vmovapd encoding.
      if (opt == llvm::CodeGenOptLevel::None) {
        EXPECT_THAT(
            store_avx.assembly,
            ::testing::ContainsRegex("vmovdqa(32|64)?[[:space:]]+%xmm[0-9]+,"));
      }
      EXPECT_THAT(store_avx.assembly,
                  ::testing::ContainsRegex(
                      "vmov(dqa(32|64)?|ap[sd])[[:space:]]+%xmm[0-9]+, 16\\("));
      EXPECT_THAT(store_avx.assembly,
                  ::testing::ContainsRegex(
                      "vmov(dqa(32|64)?|ap[sd])[[:space:]]+%xmm[0-9]+, 32\\("));
      EXPECT_THAT(store_avx.assembly,
                  ::testing::Not(::testing::ContainsRegex(
                      "vmov(dqu(8|16|32|64)?|up[sd])[[:space:]]+%xmm[0-9]+,")));
      EXPECT_THAT(store_avx.assembly,
                  ::testing::Not(::testing::HasSubstr("cmpxchg16b")));

      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts rmw_avx,
          CompileAtomic128ForTarget(kCpuAtomic128RmwMlir, kTriple, cpu,
                                    features, opt));
      ExpectNoAtomicRuntimeHelpers(rmw_avx);
      EXPECT_THAT(RunFileCheck(rmw_avx.llvm_ir, R"(
        CHECK: load atomic i128, ptr %[[I_ADDR:.*]] unordered, align 16
        CHECK: cmpxchg ptr %[[I_ADDR]], i128 %{{.*}}, i128 %{{.*}} monotonic monotonic, align 16
        CHECK: load atomic i128, ptr %[[C_ADDR:.*]] unordered, align 16
        CHECK: cmpxchg ptr %[[C_ADDR]], i128 %{{.*}}, i128 %{{.*}} monotonic monotonic, align 16
      )"),
                  absl_testing::IsOkAndHolds(true));
      EXPECT_THAT(rmw_avx.assembly, ::testing::HasSubstr("vmovdqa"));
      EXPECT_THAT(rmw_avx.assembly,
                  ::testing::ContainsRegex("lock[[:space:]]+cmpxchg16b"));
    }
  }

  // x86_64 with -cx16 (with or without +avx): cleanly rejected during lowering
  // rather than emitting __atomic_*_16 libcalls.
  for (absl::string_view features :
       {absl::string_view("-cx16"), absl::string_view("-cx16,+avx")}) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<llvm::TargetMachine> tm_no_cx16,
        CreateCrossTargetMachine(kTriple, "haswell", features,
                                 llvm::CodeGenOptLevel::Default));
    EXPECT_THAT(
        LowerAtomic128ForTarget(kCpuAtomic128StoreMlir, *tm_no_cx16),
        absl_testing::StatusIs(
            absl::StatusCode::kUnknown,
            ::testing::HasSubstr("failed to lower to LLVM atomic operation")));
    EXPECT_THAT(
        LowerAtomic128ForTarget(kCpuAtomic128RmwMlir, *tm_no_cx16),
        absl_testing::StatusIs(
            absl::StatusCode::kUnknown,
            ::testing::HasSubstr("failed to lower to LLVM atomic operation")));
  }
}

TEST_F(CpuFusionEmitterTest, Atomic128AArch64Codegen) {
  constexpr absl::string_view kTriple = "aarch64-unknown-linux-gnu";

  // Base AArch64 (no +lse, no +lse2, no +outline-atomics): lowers via native
  // ldxp/stxp loops with zero helper calls.
  for (llvm::CodeGenOptLevel opt :
       {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
    ASSERT_OK_AND_ASSIGN(
        CompiledAtomic128Artifacts store_base,
        CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, "generic",
                                  "", opt));
    ExpectNoAtomicRuntimeHelpers(store_base);
    EXPECT_THAT(store_base.assembly, ::testing::HasSubstr("ldxp"));
    EXPECT_THAT(store_base.assembly, ::testing::HasSubstr("stxp"));
    EXPECT_THAT(store_base.assembly,
                ::testing::Not(::testing::HasSubstr("casp")));

    ASSERT_OK_AND_ASSIGN(
        CompiledAtomic128Artifacts rmw_base,
        CompileAtomic128ForTarget(kCpuAtomic128RmwMlir, kTriple, "generic", "",
                                  opt));
    ExpectNoAtomicRuntimeHelpers(rmw_base);
    EXPECT_THAT(rmw_base.assembly, ::testing::HasSubstr("ldxp"));
    EXPECT_THAT(rmw_base.assembly, ::testing::HasSubstr("stxp"));
    EXPECT_THAT(rmw_base.assembly,
                ::testing::Not(::testing::HasSubstr("casp")));
  }

  // AArch64 with +lse (including +lse,+outline-atomics and neoverse-n1): lowers
  // inline to casp with zero __aarch64_cas16_* calls.
  for (const auto& [cpu, features] :
       std::vector<std::pair<absl::string_view, absl::string_view>>{
           {"generic", "+lse"},
           {"generic", "+lse,+outline-atomics"},
           {"neoverse-n1", ""}}) {
    for (llvm::CodeGenOptLevel opt :
         {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts store_lse,
          CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, cpu,
                                    features, opt));
      ExpectNoAtomicRuntimeHelpers(store_lse);
      EXPECT_THAT(store_lse.assembly, ::testing::HasSubstr("casp"));
      EXPECT_THAT(store_lse.assembly,
                  ::testing::Not(::testing::HasSubstr("ldxp")));
      EXPECT_THAT(store_lse.assembly,
                  ::testing::Not(::testing::HasSubstr("stxp")));

      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts rmw_lse,
          CompileAtomic128ForTarget(kCpuAtomic128RmwMlir, kTriple, cpu,
                                    features, opt));
      ExpectNoAtomicRuntimeHelpers(rmw_lse);
      EXPECT_THAT(rmw_lse.assembly, ::testing::HasSubstr("casp"));
      EXPECT_THAT(rmw_lse.assembly,
                  ::testing::Not(::testing::HasSubstr("ldxp")));
      EXPECT_THAT(rmw_lse.assembly,
                  ::testing::Not(::testing::HasSubstr("stxp")));
    }
  }

  // AArch64 with +lse,+lse2 (including +outline-atomics and neoverse-v2):
  // atomic_store lowers to single copy atomic stp (no casp or ldxp/stxp), and
  // atomic_rmw uses ldp for the initial load plus casp for the CAS update.
  for (const auto& [cpu, features] :
       std::vector<std::pair<absl::string_view, absl::string_view>>{
           {"generic", "+lse,+lse2"},
           {"generic", "+lse,+lse2,+outline-atomics"},
           {"neoverse-v2", ""}}) {
    for (llvm::CodeGenOptLevel opt :
         {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts store_lse2,
          CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, cpu,
                                    features, opt));
      ExpectNoAtomicRuntimeHelpers(store_lse2);
      EXPECT_THAT(RunFileCheck(store_lse2.llvm_ir, R"(
        CHECK-NOT: cmpxchg
        CHECK: store atomic i128 %{{.*}}, ptr %{{.*}} unordered, align 16
        CHECK-NOT: cmpxchg
        CHECK: store atomic i128 %{{.*}}, ptr %{{.*}} unordered, align 16
        CHECK-NOT: cmpxchg
      )"),
                  absl_testing::IsOkAndHolds(true));
      EXPECT_THAT(store_lse2.assembly, ::testing::HasSubstr("stp"));
      EXPECT_THAT(store_lse2.assembly,
                  ::testing::Not(::testing::HasSubstr("casp")));
      EXPECT_THAT(store_lse2.assembly,
                  ::testing::Not(::testing::HasSubstr("ldxp")));
      EXPECT_THAT(store_lse2.assembly,
                  ::testing::Not(::testing::HasSubstr("stxp")));

      ASSERT_OK_AND_ASSIGN(
          CompiledAtomic128Artifacts rmw_lse2,
          CompileAtomic128ForTarget(kCpuAtomic128RmwMlir, kTriple, cpu,
                                    features, opt));
      ExpectNoAtomicRuntimeHelpers(rmw_lse2);
      EXPECT_THAT(rmw_lse2.assembly, ::testing::HasSubstr("ldp"));
      EXPECT_THAT(rmw_lse2.assembly, ::testing::HasSubstr("casp"));
      EXPECT_THAT(rmw_lse2.assembly,
                  ::testing::Not(::testing::HasSubstr("ldxp")));
      EXPECT_THAT(rmw_lse2.assembly,
                  ::testing::Not(::testing::HasSubstr("stxp")));
    }
  }

  // AArch64 with +outline-atomics and without +lse/+lse2: cleanly rejected so
  // no unresolved __aarch64_cas16_* symbol can be emitted.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<llvm::TargetMachine> tm_outline,
      CreateCrossTargetMachine(kTriple, "generic", "+outline-atomics",
                               llvm::CodeGenOptLevel::Default));
  EXPECT_THAT(
      LowerAtomic128ForTarget(kCpuAtomic128StoreMlir, *tm_outline),
      absl_testing::StatusIs(
          absl::StatusCode::kUnknown,
          ::testing::HasSubstr("failed to lower to LLVM atomic operation")));
  EXPECT_THAT(
      LowerAtomic128ForTarget(kCpuAtomic128RmwMlir, *tm_outline),
      absl_testing::StatusIs(
          absl::StatusCode::kUnknown,
          ::testing::HasSubstr("failed to lower to LLVM atomic operation")));

  // AArch64 with +lse2,+outline-atomics (no +lse): direct store lowers to
  // native stp without helper calls, while RMW is cleanly rejected.
  for (llvm::CodeGenOptLevel opt :
       {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Default}) {
    ASSERT_OK_AND_ASSIGN(
        CompiledAtomic128Artifacts store_lse2_outline,
        CompileAtomic128ForTarget(kCpuAtomic128StoreMlir, kTriple, "generic",
                                  "+lse2,+outline-atomics", opt));
    ExpectNoAtomicRuntimeHelpers(store_lse2_outline);
    EXPECT_THAT(store_lse2_outline.assembly, ::testing::HasSubstr("stp"));
    EXPECT_THAT(store_lse2_outline.assembly,
                ::testing::Not(::testing::HasSubstr("casp")));

    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<llvm::TargetMachine> tm_lse2_outline,
        CreateCrossTargetMachine(kTriple, "generic", "+lse2,+outline-atomics",
                                 opt));
    EXPECT_THAT(
        LowerAtomic128ForTarget(kCpuAtomic128RmwMlir, *tm_lse2_outline),
        absl_testing::StatusIs(
            absl::StatusCode::kUnknown,
            ::testing::HasSubstr("failed to lower to LLVM atomic operation")));
  }
}

TEST_F(CpuFusionEmitterTest, Atomic128RejectsUnderalignedC128OnCapableCpu) {
  constexpr absl::string_view kUnderalignedStore = R"(
    module attributes {xla.cpu_memory_region_name = "c128_align8_store"} {
      func.func @c128_align8_store(
          %buf: tensor<4xcomplex<f64>> {llvm.align = 8 : index, xla.slice_index = 0})
          -> tensor<4xcomplex<f64>>
          attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
        %c0 = arith.constant 0 : index
        %v = complex.constant [1.0 : f64, 2.0 : f64] : complex<f64>
        %out = xla.atomic_store %v into %buf[%c0] : tensor<4xcomplex<f64>>
        return %out : tensor<4xcomplex<f64>>
      }
    }
  )";
  constexpr absl::string_view kUnderalignedRmw = R"(
    module attributes {xla.cpu_memory_region_name = "c128_align8_rmw"} {
      func.func @c128_align8_rmw(
          %buf: tensor<4xcomplex<f64>> {llvm.align = 8 : index, xla.slice_index = 0})
          -> tensor<4xcomplex<f64>>
          attributes {xla.backend_kind = #xla.backend_kind<cpu>, xla.entry} {
        %c0 = arith.constant 0 : index
        %v = complex.constant [1.0 : f64, 2.0 : f64] : complex<f64>
        %out = xla.atomic_rmw %buf[%c0] : tensor<4xcomplex<f64>> {
          ^bb0(%cur: complex<f64>):
            %sum = complex.add %cur, %v : complex<f64>
            xla.yield %sum : complex<f64>
        }
        return %out : tensor<4xcomplex<f64>>
      }
    }
  )";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<llvm::TargetMachine> tm,
      CreateCrossTargetMachine("x86_64-unknown-linux-gnu", "haswell", "",
                               llvm::CodeGenOptLevel::Default));
  EXPECT_THAT(
      LowerAtomic128ForTarget(kUnderalignedStore, *tm),
      absl_testing::StatusIs(
          absl::StatusCode::kUnknown,
          ::testing::HasSubstr("failed to lower to LLVM atomic operation")));
  EXPECT_THAT(
      LowerAtomic128ForTarget(kUnderalignedRmw, *tm),
      absl_testing::StatusIs(
          absl::StatusCode::kUnknown,
          ::testing::HasSubstr("failed to lower to LLVM atomic operation")));
}

TEST_F(CpuFusionEmitterTest, Atomic128HostJitExecutesC128StoreAndAdd) {
  std::unique_ptr<IrCompiler> ir_compiler =
      IrCompiler::Create({},
                         IrCompiler::Options{
                             /*opt_level=*/llvm::CodeGenOptLevel::Default,
                             /*optimize_for_size=*/false,
                             TargetMachineOptions(GetDebugOptionsFromFlags()),
                         },
                         IrCompiler::CompilationHooks());
  ASSERT_OK_AND_ASSIGN(JitCompiler jit,
                       JitCompiler::Create({}, std::move(ir_compiler)));
  CpuDeviceSpec host_spec =
      CpuDeviceSpecFromTargetMachine(jit.target_machine());
  if (!host_spec.has_atomic_cmpxchg_128) {
    GTEST_SKIP() << "Host CPU does not support native 128 bit atomic CAS.";
  }

  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource store_llvm,
      LowerAtomic128ForTarget(kCpuAtomic128StoreMlir, *jit.target_machine()));
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource rmw_llvm,
      LowerAtomic128ForTarget(kCpuAtomic128RmwMlir, *jit.target_machine()));
  ASSERT_OK(jit.AddModule(std::move(store_llvm).thread_safe_module()));
  ASSERT_OK(jit.AddModule(std::move(rmw_llvm).thread_safe_module()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionLibrary> library,
      std::move(jit).Compile({FunctionLibrary::Sym<FunctionLibrary::Kernel>(
                                  "cpu_atomic_128_store"),
                              FunctionLibrary::Sym<FunctionLibrary::Kernel>(
                                  "cpu_atomic_128_rmw")}));
  ASSERT_OK_AND_ASSIGN(FunctionLibrary::Kernel * store_kernel,
                       library->ResolveFunction<FunctionLibrary::Kernel>(
                           "cpu_atomic_128_store"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * rmw_kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("cpu_atomic_128_rmw"));

  struct alignas(16) Uint128Words {
    uint64_t lo;
    uint64_t hi;
  };
  static_assert(sizeof(Uint128Words) == 16 && alignof(Uint128Words) == 16);

  alignas(16) std::array<Uint128Words, 4> i128_buf = {
      Uint128Words{1, 2}, Uint128Words{3, 4}, Uint128Words{5, 6},
      Uint128Words{7, 8}};
  alignas(16) Uint128Words i128_store_val{0xffffffffffffffffULL, 0x10ULL};
  alignas(16) std::array<complex128, 4> c128_buf = {
      complex128(1.0, 2.0), complex128(3.0, 4.0), complex128(5.0, 6.0),
      complex128(7.0, 8.0)};
  alignas(16) complex128 c128_store_val(12.5, -3.25);

  std::array<XLA_CPU_KernelArg, 4> store_args = {{
      {i128_buf.data(), sizeof(i128_buf)},
      {&i128_store_val, sizeof(i128_store_val)},
      {c128_buf.data(), sizeof(c128_buf)},
      {&c128_store_val, sizeof(c128_store_val)},
  }};
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame store_frame = {&num_workgroups, &workgroup_id,
                                         store_args.size(), store_args.data()};
  ASSERT_EQ(CallCpuKernel(store_kernel, &store_frame), nullptr);

  EXPECT_EQ(i128_buf[0].lo, 1u);
  EXPECT_EQ(i128_buf[0].hi, 2u);
  EXPECT_EQ(i128_buf[1].lo, 0xffffffffffffffffULL);
  EXPECT_EQ(i128_buf[1].hi, 0x10ULL);
  EXPECT_EQ(c128_buf[1], complex128(3.0, 4.0));
  EXPECT_EQ(c128_buf[2], complex128(12.5, -3.25));

  // Verify multi threaded atomic add across both 64 bit halves (including carry
  // propagation across bit 63 for i128 and exact accumulation for c128).
  constexpr int kNumThreads = 8;
  constexpr int kItersPerThread = 500;
  alignas(16) Uint128Words i128_delta{1ULL, 2ULL};
  alignas(16) complex128 c128_delta(1.5, -0.5);
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "atomic128_pool",
                                 kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
      pool.Schedule([&]() {
        std::array<XLA_CPU_KernelArg, 4> rmw_args = {{
            {i128_buf.data(), sizeof(i128_buf)},
            {&i128_delta, sizeof(i128_delta)},
            {c128_buf.data(), sizeof(c128_buf)},
            {&c128_delta, sizeof(c128_delta)},
        }};
        XLA_CPU_NumWorkGroups nwg = {1, 1, 1};
        XLA_CPU_WorkGroupId wgid = {0, 0, 0};
        XLA_CPU_KernelCallFrame rmw_frame = {&nwg, &wgid, rmw_args.size(),
                                             rmw_args.data()};
        for (int i = 0; i < kItersPerThread; ++i) {
          EXPECT_EQ(CallCpuKernel(rmw_kernel, &rmw_frame), nullptr);
        }
      });
    }
  }

  constexpr uint64_t kTotalUpdates = kNumThreads * kItersPerThread;
  // Initial i128_buf[1] was (2^64 - 1) + 0x10 * 2^64; adding (1 + 2 * 2^64)
  // kTotalUpdates times carries 1 into the high 64 bit word.
  EXPECT_EQ(i128_buf[1].lo, kTotalUpdates - 1);
  EXPECT_EQ(i128_buf[1].hi, 0x10ULL + 1ULL + 2ULL * kTotalUpdates);
  EXPECT_EQ(c128_buf[2], complex128(12.5 + 1.5 * kTotalUpdates,
                                    -3.25 - 0.5 * kTotalUpdates));
  // Neighboring elements must be untouched.
  EXPECT_EQ(i128_buf[0].lo, 1u);
  EXPECT_EQ(i128_buf[2].lo, 5u);
  EXPECT_EQ(c128_buf[1], complex128(3.0, 4.0));
  EXPECT_EQ(c128_buf[3], complex128(7.0, 8.0));
}

}  // namespace
}  // namespace cpu
}  // namespace xla
