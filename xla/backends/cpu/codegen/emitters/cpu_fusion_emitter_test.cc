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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/emitters/cpu_scatter_emitter.h"
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/fusion_emitter.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_attrs.h.inc"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace cpu {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

struct AlignedBufferDeleter {
  void operator()(void* ptr) const { std::free(ptr); }
};

template <typename T>
absl::StatusOr<std::unique_ptr<T[], AlignedBufferDeleter>>
AllocateAlignedBuffer(size_t num_elements, T initial_value = T{}) {
  size_t alignment = std::max<size_t>(MinAlign(), alignof(T));
  size_t byte_size = std::max<size_t>(alignment, num_elements * sizeof(T));
  byte_size = RoundUpTo<size_t>(byte_size, alignment);
  void* raw = std::aligned_alloc(alignment, byte_size);
  if (raw == nullptr) {
    return absl::ResourceExhaustedError("aligned_alloc failed");
  }
  T* typed = static_cast<T*>(raw);
  for (size_t i = 0; i < num_elements; ++i) {
    typed[i] = initial_value;
  }
  return std::unique_ptr<T[], AlignedBufferDeleter>(typed);
}

bool ScatterConstraintsSatisfied(const IndexingMap& map, int64_t thread_id,
                                 int64_t index_id, int64_t vector_id,
                                 int64_t vector_element_id) {
  mlir::MLIRContext* ctx = map.GetMLIRContext();
  return map.ConstraintsSatisfied(
      {CreateSymbolicConstant(thread_id, ctx)},
      {CreateSymbolicConstant(index_id, ctx),
       CreateSymbolicConstant(vector_id, ctx),
       CreateSymbolicConstant(vector_element_id, ctx)});
}

llvm::SmallVector<int64_t, 4> EvaluateScatterMap(const IndexingMap& map,
                                                 int64_t thread_id,
                                                 int64_t index_id,
                                                 int64_t vector_id,
                                                 int64_t vector_element_id) {
  mlir::MLIRContext* ctx = map.GetMLIRContext();
  return map.Evaluate({CreateSymbolicConstant(thread_id, ctx)},
                      {CreateSymbolicConstant(index_id, ctx),
                       CreateSymbolicConstant(vector_id, ctx),
                       CreateSymbolicConstant(vector_element_id, ctx)});
}

// Builds a module whose entry function iterates over `map` and increments each
// visited element of a padded count tensor. The padding keeps out of bounds
// accesses (such as `updates_id == num_updates`) inside the buffer so tests can
// observe them directly without undefined behavior.
mlir::OwningOpRef<mlir::ModuleOp> BuildScatterLoopModule(
    const IndexingMap& map, absl::Span<const int64_t> padded_shape,
    int64_t num_threads, mlir::MLIRContext* ctx) {
  mlir::OpBuilder builder(ctx);
  mlir::Location loc = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(loc, "scatter_loop_module");

  mlir::DataLayoutEntryInterface index_layout = mlir::DataLayoutEntryAttr::get(
      builder.getIndexType(), builder.getI32IntegerAttr(64));
  module->getOperation()->setAttr(
      mlir::DLTIDialect::kDataLayoutAttrName,
      mlir::DataLayoutSpecAttr::get(ctx, {index_layout}));
  module->getOperation()->setAttr(xla::CpuMemoryRegionNameAttr::name,
                                  builder.getStringAttr("scatter_loop_test"));

  mlir::RankedTensorType tensor_type =
      mlir::RankedTensorType::get(padded_shape, builder.getI32Type());
  builder.setInsertionPointToStart(module->getBody());
  auto func = mlir::func::FuncOp::create(
      builder, loc, "visit_scatter",
      builder.getFunctionType({tensor_type}, {tensor_type}));
  func->setAttr("xla.entry", builder.getUnitAttr());
  SetBackendKind(ctx, func, BackendKind::kCpu);

  int64_t byte_size =
      Product(padded_shape) * static_cast<int64_t>(sizeof(int32_t));
  func.setArgAttrs(
      0, {builder.getNamedAttr("xla.slice_index", builder.getIndexAttr(0)),
          builder.getNamedAttr("xla.aliased", builder.getUnitAttr()),
          builder.getNamedAttr(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                               builder.getIndexAttr(MinAlign())),
          builder.getNamedAttr(
              mlir::LLVM::LLVMDialect::getDereferenceableAttrName(),
              builder.getIndexAttr(byte_size))});

  mlir::ImplicitLocOpBuilder b(loc, func);
  b.setInsertionPointToStart(func.addEntryBlock());
  WorkGroupIdOp workgroup_id = WorkGroupIdOp::create(b, WorkGroupDimension::x);
  workgroup_id->setAttr("xla.range", b.getIndexArrayAttr({0, num_threads - 1}));

  mlir::ValueRange results = emitters::EmitXlaLoopOp(
      b, {workgroup_id}, {func.getArgument(0)}, map,
      [&](mlir::ImplicitLocOpBuilder& nested_b, mlir::ValueRange ivs,
          mlir::ValueRange map_results,
          mlir::ValueRange iter_args) -> llvm::SmallVector<mlir::Value> {
        mlir::Value cur = mlir::tensor::ExtractOp::create(
            nested_b, iter_args[0], map_results);
        mlir::Value one =
            mlir::arith::ConstantIntOp::create(nested_b, 1, /*width=*/32);
        mlir::Value next = mlir::arith::AddIOp::create(nested_b, cur, one);
        mlir::Value updated = mlir::tensor::InsertOp::create(
            nested_b, next, iter_args[0], map_results);
        return {updated};
      });
  mlir::func::ReturnOp::create(b, results);
  return module;
}

struct LoweredScatterLoopIr {
  std::string after_peel;
  std::string after_scf;
};

absl::StatusOr<LoweredScatterLoopIr> RunPeelAndLowerScatterLoop(
    const IndexingMap& map, absl::Span<const int64_t> updates_shape,
    int64_t num_threads, mlir::MLIRContext* ctx) {
  std::vector<int64_t> padded_shape;
  padded_shape.reserve(updates_shape.size());
  for (int64_t dim : updates_shape) {
    padded_shape.push_back(dim + 4);
  }
  mlir::OwningOpRef<mlir::ModuleOp> module =
      BuildScatterLoopModule(map, padded_shape, num_threads, ctx);

  LoweredScatterLoopIr out;
  {
    mlir::PassManager peel_pm(ctx);
    peel_pm.enableVerifier(true);
    peel_pm.addNestedPass<mlir::func::FuncOp>(
        cpu::createPeelWorkgroupLoopPass());
    tsl::StatusScopedDiagnosticHandler diag(ctx);
    ABSL_RETURN_IF_ERROR(diag.consumeStatus(peel_pm.run(*module)));
  }
  out.after_peel = mlir::debugString(*module);

  {
    mlir::PassManager scf_pm(ctx);
    scf_pm.enableVerifier(true);
    scf_pm.addNestedPass<mlir::func::FuncOp>(
        emitters::createLowerXlaLoopsToScfPass());
    tsl::StatusScopedDiagnosticHandler diag(ctx);
    ABSL_RETURN_IF_ERROR(diag.consumeStatus(scf_pm.run(*module)));
  }
  out.after_scf = mlir::debugString(*module);
  return out;
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>> JitCompileKernel(
    LlvmKernelSource llvm_source, absl::string_view kernel_name) {
  std::unique_ptr<IrCompiler> ir_compiler =
      IrCompiler::Create({},
                         IrCompiler::Options{
                             /*opt_level=*/llvm::CodeGenOptLevel::Default,
                             /*optimize_for_size=*/false,
                             TargetMachineOptions(GetDebugOptionsFromFlags()),
                         },
                         IrCompiler::CompilationHooks());
  ABSL_ASSIGN_OR_RETURN(JitCompiler jit,
                        JitCompiler::Create({}, std::move(ir_compiler)));
  ABSL_RETURN_IF_ERROR(
      jit.AddModule(std::move(llvm_source).thread_safe_module()));
  return std::move(jit).Compile(
      {FunctionLibrary::Sym<FunctionLibrary::Kernel>(kernel_name)});
}

ABSL_ATTRIBUTE_NO_SANITIZE_CFI XLA_CPU_KernelError* InvokeKernel(
    FunctionLibrary::Kernel* kernel, XLA_CPU_KernelCallFrame* frame) {
  return (*kernel)(frame);
}

// Compiles a `GetScatterIndexingMap` loop through the full CPU FusionCompiler
// pipeline (including PeelWorkgroupLoopPass and LowerXlaLoopsToScfPass), runs
// all `num_threads` workgroups against a padded buffer, and checks that every
// in bounds update index is visited once and no out of bounds index (including
// `updates_id == num_updates`) is touched.
absl::Status VerifyScatterLoopExecution(absl::Span<const int64_t> updates_shape,
                                        int64_t num_threads,
                                        int64_t vector_size) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();
  IndexingMap map =
      GetScatterIndexingMap(updates_shape, num_threads, vector_size, ctx.get());
  map.Simplify();

  constexpr int64_t kGuardPadding = 4;
  std::vector<int64_t> padded_shape;
  padded_shape.reserve(updates_shape.size());
  for (int64_t dim : updates_shape) {
    padded_shape.push_back(dim + num_threads + vector_size + kGuardPadding);
  }
  int64_t total_padded = Product(padded_shape);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      BuildScatterLoopModule(map, padded_shape, num_threads, ctx.get());
  FusionCompiler compiler(ctx.get(), FusionCompiler::Options{512, 1, true});
  ABSL_ASSIGN_OR_RETURN(LlvmKernelSource llvm_source,
                        compiler.Compile(MlirKernelSource(std::move(module))));
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<FunctionLibrary> library,
      JitCompileKernel(std::move(llvm_source), "visit_scatter"));
  ABSL_ASSIGN_OR_RETURN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("visit_scatter"));

  ABSL_ASSIGN_OR_RETURN(auto counts,
                        AllocateAlignedBuffer<int32_t>(total_padded, 0));
  XLA_CPU_KernelArg arg = {counts.get(),
                           static_cast<size_t>(total_padded) * sizeof(int32_t)};
  XLA_CPU_NumWorkGroups num_workgroups = {static_cast<uint64_t>(num_threads), 1,
                                          1};
  for (int64_t wg = 0; wg < num_threads; ++wg) {
    XLA_CPU_WorkGroupId workgroup_id = {static_cast<uint64_t>(wg), 0, 0};
    XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, 1, &arg};
    if (InvokeKernel(kernel, &frame) != nullptr) {
      return absl::InternalError("visit_scatter returned a kernel error");
    }
  }

  bool is_empty_domain = Product(updates_shape) == 0;
  for (int64_t linear = 0; linear < total_padded; ++linear) {
    int64_t rem = linear;
    bool in_bounds = !is_empty_domain;
    for (int64_t d = static_cast<int64_t>(padded_shape.size()) - 1; d >= 0;
         --d) {
      int64_t coord = rem % padded_shape[d];
      rem /= padded_shape[d];
      if (coord >= updates_shape[d]) {
        in_bounds = false;
      }
    }
    int32_t expected = in_bounds ? 1 : 0;
    if (counts[linear] != expected) {
      return absl::InternalError(absl::StrCat("count mismatch at linear index ",
                                              linear, ": got ", counts[linear],
                                              ", expected ", expected));
    }
  }
  return absl::OkStatus();
}

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

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapNonDivisibleNumUpdates) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  // num_updates = 5, num_threads = 3 => num_updates_per_thread = 2.
  // Workgroup 2 has index_id 0 -> updates_id 4 (last valid update) and
  // index_id 1 -> updates_id 5 (must be rejected).
  IndexingMap map = GetScatterIndexingMap(
      /*updates_operand_shape=*/{5, 4}, /*num_threads=*/3, /*vector_size=*/4,
      ctx.get());
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, 2}));
  EXPECT_THAT(map.GetSymbolBounds(),
              ElementsAre(Interval{0, 1}, Interval{0, 0}, Interval{0, 3}));
  EXPECT_TRUE(ScatterConstraintsSatisfied(map, /*thread_id=*/0, /*index_id=*/0,
                                          /*vector_id=*/0,
                                          /*vector_element_id=*/0));
  EXPECT_TRUE(ScatterConstraintsSatisfied(map, /*thread_id=*/1, /*index_id=*/1,
                                          /*vector_id=*/0,
                                          /*vector_element_id=*/3));
  EXPECT_EQ(EvaluateScatterMap(map, 2, 0, 0, 3),
            (llvm::SmallVector<int64_t, 4>{4, 3}));
  EXPECT_TRUE(ScatterConstraintsSatisfied(map, /*thread_id=*/2, /*index_id=*/0,
                                          /*vector_id=*/0,
                                          /*vector_element_id=*/3));
  EXPECT_EQ(EvaluateScatterMap(map, 2, 1, 0, 0),
            (llvm::SmallVector<int64_t, 4>{5, 0}));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map, /*thread_id=*/2, /*index_id=*/1,
                                           /*vector_id=*/0,
                                           /*vector_element_id=*/0));

  map.Simplify();
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, 2}));
  EXPECT_THAT(map.GetSymbolBounds(),
              ElementsAre(Interval{0, 1}, Interval{0, 0}, Interval{0, 3}));
  ASSERT_EQ(map.GetConstraintsCount(), 1);
  EXPECT_EQ(map.GetSymbolicConstraints().front().second, (Interval{0, 4}));
  EXPECT_THAT(map.ComputeResultRanges(),
              ElementsAre(Interval{0, 4}, Interval{0, 3}));
  EXPECT_TRUE(ScatterConstraintsSatisfied(map, 2, 0, 0, 0));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map, 2, 1, 0, 0));

  // PeelWorkgroupLoopPass peels workgroups 0..1 without constraints and
  // narrows workgroup 2's index_id domain to [0, 0] (updates_id == 4 only).
  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr ir,
      RunPeelAndLowerScatterLoop(map, {5, 4}, /*num_threads=*/3, ctx.get()));
  EXPECT_THAT(RunFileCheck(ir.after_peel, R"(
    CHECK: scf.if
    CHECK:   xla.loop {{.*}} in #xla.indexing_map<"(thread_id)[index_id, vector_id, vector_element_id] -> (thread_id * 2 + index_id, vector_element_id), domain: thread_id in [0, 1], index_id in [0, 1], vector_id in [0, 0], vector_element_id in [0, 3]">
    CHECK: else
    CHECK:   xla.loop {{.*}} in #xla.indexing_map<"(thread_id)[index_id, vector_id, vector_element_id] -> (4, vector_element_id), domain: thread_id in [2, 2], index_id in [0, 0], vector_id in [0, 0], vector_element_id in [0, 3]">
  )"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(ir.after_scf, R"(
    CHECK-NOT: xla.loop
    CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
    CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
    CHECK:     scf.if
    CHECK:       scf.for %{{.+}} = %{{.+}} to %[[C2]]
    CHECK:     else
    CHECK:       scf.for %{{.+}} = %{{.+}} to %[[C1]]
  )"),
              absl_testing::IsOkAndHolds(true));

  // Two threads with 5 updates (3 updates per thread): workgroup 1 checks
  // `updates_id <= 4` directly in SCF without peeling, rejecting 5.
  IndexingMap two_thread_map = GetScatterIndexingMap(
      {5, 4}, /*num_threads=*/2, /*vector_size=*/4, ctx.get());
  EXPECT_TRUE(ScatterConstraintsSatisfied(two_thread_map, 1, 1, 0, 0));
  EXPECT_FALSE(ScatterConstraintsSatisfied(two_thread_map, 1, 2, 0, 0));
  two_thread_map.Simplify();
  ASSERT_EQ(two_thread_map.GetConstraintsCount(), 1);
  EXPECT_EQ(two_thread_map.GetSymbolicConstraints().front().second,
            (Interval{0, 4}));
  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr two_thread_ir,
      RunPeelAndLowerScatterLoop(two_thread_map, {5, 4}, /*num_threads=*/2,
                                 ctx.get()));
  EXPECT_THAT(RunFileCheck(two_thread_ir.after_scf, R"(
    CHECK-NOT: xla.loop
    CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
    CHECK:     scf.for
    CHECK:       arith.cmpi sle, %{{.+}}, %[[C4]]
  )"),
              absl_testing::IsOkAndHolds(true));

  EXPECT_OK(VerifyScatterLoopExecution({5, 4}, /*num_threads=*/2,
                                       /*vector_size=*/4));
  EXPECT_OK(VerifyScatterLoopExecution({5, 4}, /*num_threads=*/3,
                                       /*vector_size=*/4));
  // Includes both a partial workgroup (2) and a fully idle workgroup (3).
  EXPECT_OK(VerifyScatterLoopExecution({5, 4}, /*num_threads=*/4,
                                       /*vector_size=*/4));
  // Even updates with an idle trailing workgroup (3), where updates_id == 6
  // is the first index of workgroup 3.
  EXPECT_OK(VerifyScatterLoopExecution({6, 4}, /*num_threads=*/4,
                                       /*vector_size=*/4));
  // Multi update partial tail: workgroup 2 handles updates 8 and 9, skips 10
  // and 11.
  EXPECT_OK(VerifyScatterLoopExecution({10, 4}, /*num_threads=*/3,
                                       /*vector_size=*/4));
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapMoreThreadsThanUpdates) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  IndexingMap map = GetScatterIndexingMap(
      /*updates_operand_shape=*/{3, 4}, /*num_threads=*/8, /*vector_size=*/4,
      ctx.get());
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, 7}));
  EXPECT_THAT(map.GetSymbolBounds(),
              ElementsAre(Interval{0, 0}, Interval{0, 0}, Interval{0, 3}));
  for (int64_t tid = 0; tid < 3; ++tid) {
    EXPECT_EQ(EvaluateScatterMap(map, tid, 0, 0, 0),
              (llvm::SmallVector<int64_t, 4>{tid, 0}));
    EXPECT_TRUE(ScatterConstraintsSatisfied(map, tid, 0, 0, 0));
  }
  // updates_id == num_updates (tid == 3) and all higher threads are rejected.
  for (int64_t tid = 3; tid < 8; ++tid) {
    EXPECT_FALSE(ScatterConstraintsSatisfied(map, tid, 0, 0, 0));
  }

  map.Simplify();
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, 2}));
  EXPECT_THAT(map.GetSymbolBounds(),
              ElementsAre(Interval{0, 0}, Interval{0, 0}, Interval{0, 3}));
  EXPECT_THAT(map.GetSymbolicConstraints(), IsEmpty());
  EXPECT_THAT(map.ComputeResultRanges(),
              ElementsAre(Interval{0, 2}, Interval{0, 3}));

  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr ir,
      RunPeelAndLowerScatterLoop(map, {3, 4}, /*num_threads=*/8, ctx.get()));
  EXPECT_THAT(RunFileCheck(ir.after_scf, R"(
    CHECK-NOT: xla.loop
    CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
    CHECK-DAG: %[[WG:.+]] = xla.workgroup_id x
    CHECK:     scf.for
    CHECK:       arith.cmpi sle, %[[WG]], %[[C2]]
  )"),
              absl_testing::IsOkAndHolds(true));

  EXPECT_OK(VerifyScatterLoopExecution({3, 4}, /*num_threads=*/8,
                                       /*vector_size=*/4));
  EXPECT_OK(VerifyScatterLoopExecution({1, 4}, /*num_threads=*/4,
                                       /*vector_size=*/4));
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapNonDivisibleVectorSize) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  // num_slice_elements = 3 * 2 = 6, vector_size = 4 => 2 vectors per slice;
  // (vector_id=1, vector_element_id=1) -> linear 5 is valid, linear 6 and 7
  // are rejected.
  IndexingMap map = GetScatterIndexingMap(
      /*updates_operand_shape=*/{5, 3, 2}, /*num_threads=*/3,
      /*vector_size=*/4, ctx.get());
  EXPECT_FALSE(map.IsKnownEmpty());
  EXPECT_THAT(map.GetSymbolBounds(),
              ElementsAre(Interval{0, 1}, Interval{0, 1}, Interval{0, 3}));
  EXPECT_EQ(EvaluateScatterMap(map, 2, 0, 1, 1),
            (llvm::SmallVector<int64_t, 4>{4, 2, 1}));
  EXPECT_TRUE(ScatterConstraintsSatisfied(map, 2, 0, 1, 1));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map, 0, 0, 1, 2));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map, 0, 0, 1, 3));
  EXPECT_FALSE(ScatterConstraintsSatisfied(map, 2, 1, 0, 0));

  map.Simplify();
  EXPECT_FALSE(map.IsKnownEmpty());
  ASSERT_EQ(map.GetConstraintsCount(), 2);
  std::vector<Interval> constraint_intervals;
  for (const auto& [expr, interval] : map.GetSymbolicConstraints()) {
    constraint_intervals.push_back(interval);
  }
  EXPECT_THAT(constraint_intervals,
              ::testing::UnorderedElementsAre(Interval{0, 4}, Interval{0, 5}));
  EXPECT_EQ(map.ComputeResultRanges()[0], (Interval{0, 4}));

  // PeelWorkgroupLoopPass must not strip the slice tail constraint.
  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr ir,
      RunPeelAndLowerScatterLoop(map, {5, 3, 2}, /*num_threads=*/3, ctx.get()));
  EXPECT_THAT(RunFileCheck(ir.after_peel, R"(
    CHECK-NOT: scf.if
    CHECK:     xla.loop
  )"),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(RunFileCheck(ir.after_scf, R"(
    CHECK-NOT: xla.loop
    CHECK-DAG: arith.constant 4 : index
    CHECK-DAG: arith.constant 5 : index
    CHECK:     scf.if
  )"),
              absl_testing::IsOkAndHolds(true));

  // When vector_size > num_slice_elements, Simplify folds the slice constraint
  // into vector_element_id's domain, enabling workgroup peeling on updates_id.
  IndexingMap small_slice_map = GetScatterIndexingMap(
      {5, 3}, /*num_threads=*/3, /*vector_size=*/4, ctx.get());
  EXPECT_TRUE(ScatterConstraintsSatisfied(small_slice_map, 2, 0, 0, 2));
  EXPECT_FALSE(ScatterConstraintsSatisfied(small_slice_map, 0, 0, 0, 3));
  EXPECT_FALSE(ScatterConstraintsSatisfied(small_slice_map, 2, 1, 0, 0));
  small_slice_map.Simplify();
  EXPECT_THAT(small_slice_map.GetSymbolBounds(),
              ElementsAre(Interval{0, 1}, Interval{0, 0}, Interval{0, 2}));
  ASSERT_EQ(small_slice_map.GetConstraintsCount(), 1);
  EXPECT_EQ(small_slice_map.GetSymbolicConstraints().front().second,
            (Interval{0, 4}));
  ASSERT_OK_AND_ASSIGN(
      LoweredScatterLoopIr small_slice_ir,
      RunPeelAndLowerScatterLoop(small_slice_map, {5, 3}, /*num_threads=*/3,
                                 ctx.get()));
  EXPECT_THAT(RunFileCheck(small_slice_ir.after_peel, R"(
    CHECK: scf.if
    CHECK:   xla.loop
    CHECK: else
    CHECK:   xla.loop
  )"),
              absl_testing::IsOkAndHolds(true));

  EXPECT_OK(VerifyScatterLoopExecution({5, 3, 2}, /*num_threads=*/3,
                                       /*vector_size=*/4));
  EXPECT_OK(VerifyScatterLoopExecution({5, 3}, /*num_threads=*/3,
                                       /*vector_size=*/4));
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapZeroUpdatesAndZeroSlices) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  struct EmptyCase {
    std::vector<int64_t> updates_shape;
    int64_t num_threads;
    int64_t vector_size;
  };
  const std::vector<EmptyCase> cases = {
      {{0, 4, 8}, 1, 4},  // num_updates == 0, single thread
      {{0, 4, 8}, 4, 4},  // num_updates == 0, multi thread
      {{0}, 2, 1},        // num_updates == 0, rank 0 operand
      {{5, 0, 8}, 1, 1},  // num_slice_elements == 0, single thread
      {{5, 4, 0}, 3, 4},  // num_slice_elements == 0, multi thread, vector > 1
      {{0, 0}, 2, 1},     // both num_updates == 0 and num_slice_elements == 0
  };

  for (const EmptyCase& tc : cases) {
    IndexingMap map = GetScatterIndexingMap(tc.updates_shape, tc.num_threads,
                                            tc.vector_size, ctx.get());
    EXPECT_TRUE(map.IsKnownEmpty());
    EXPECT_EQ(map.GetNumResults(),
              static_cast<int64_t>(tc.updates_shape.size()));
    EXPECT_FALSE(ScatterConstraintsSatisfied(map, 0, 0, 0, 0));

    map.Simplify();
    EXPECT_TRUE(map.IsKnownEmpty());
    EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, -1}));
    EXPECT_THAT(map.GetSymbolBounds(),
                ElementsAre(Interval{0, -1}, Interval{0, -1}, Interval{0, -1}));
    EXPECT_THAT(map.GetSymbolicConstraints(), IsEmpty());

    ASSERT_OK_AND_ASSIGN(LoweredScatterLoopIr ir,
                         RunPeelAndLowerScatterLoop(map, tc.updates_shape,
                                                    tc.num_threads, ctx.get()));
    EXPECT_THAT(RunFileCheck(ir.after_scf, R"(
      CHECK-NOT: xla.loop
      CHECK:     %[[C0:.+]] = arith.constant 0 : index
      CHECK:     scf.for %{{.+}} = %[[C0]] to %[[C0]]
    )"),
                absl_testing::IsOkAndHolds(true));
    EXPECT_OK(VerifyScatterLoopExecution(tc.updates_shape, tc.num_threads,
                                         tc.vector_size));
  }
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapRankZeroScalarOperand) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();

  // Rank 0 scatter operand has updates shape [num_updates], so slice_shape is
  // empty and num_slice_elements == 1. Test both vector_size == 1 and
  // vector_size > 1 (where vector_element_id >= 1 must be rejected so each
  // scalar update is applied once).
  for (int64_t vector_size : {int64_t{1}, int64_t{4}}) {
    IndexingMap map = GetScatterIndexingMap(
        /*updates_operand_shape=*/{5}, /*num_threads=*/3, vector_size,
        ctx.get());
    EXPECT_FALSE(map.IsKnownEmpty());
    EXPECT_EQ(map.GetNumResults(), 1);
    EXPECT_EQ(EvaluateScatterMap(map, 2, 0, 0, 0),
              (llvm::SmallVector<int64_t, 4>{4}));
    EXPECT_TRUE(ScatterConstraintsSatisfied(map, 2, 0, 0, 0));
    EXPECT_FALSE(ScatterConstraintsSatisfied(map, 2, 1, 0, 0));
    if (vector_size > 1) {
      EXPECT_FALSE(ScatterConstraintsSatisfied(map, 0, 0, 0, 1));
    }

    map.Simplify();
    EXPECT_FALSE(map.IsKnownEmpty());
    EXPECT_EQ(map.GetDimensionBound(0), (Interval{0, 2}));
    EXPECT_THAT(map.GetSymbolBounds(),
                ElementsAre(Interval{0, 1}, Interval{0, 0}, Interval{0, 0}));
    ASSERT_EQ(map.GetConstraintsCount(), 1);
    EXPECT_EQ(map.GetSymbolicConstraints().front().second, (Interval{0, 4}));
    EXPECT_THAT(map.ComputeResultRanges(), ElementsAre(Interval{0, 4}));

    ASSERT_OK_AND_ASSIGN(
        LoweredScatterLoopIr ir,
        RunPeelAndLowerScatterLoop(map, {5}, /*num_threads=*/3, ctx.get()));
    EXPECT_THAT(RunFileCheck(ir.after_peel, R"(
      CHECK: scf.if
      CHECK:   xla.loop {{.*}} in #xla.indexing_map<"(thread_id)[index_id, vector_id, vector_element_id] -> (thread_id * 2 + index_id), domain: thread_id in [0, 1], index_id in [0, 1], vector_id in [0, 0], vector_element_id in [0, 0]">
      CHECK: else
      CHECK:   xla.loop {{.*}} in #xla.indexing_map<"(thread_id)[index_id, vector_id, vector_element_id] -> (4), domain: thread_id in [2, 2], index_id in [0, 0], vector_id in [0, 0], vector_element_id in [0, 0]">
    )"),
                absl_testing::IsOkAndHolds(true));
    EXPECT_THAT(RunFileCheck(ir.after_scf, "CHECK-NOT: xla.loop"),
                absl_testing::IsOkAndHolds(true));
    EXPECT_OK(VerifyScatterLoopExecution({5}, /*num_threads=*/3, vector_size));
  }

  EXPECT_OK(
      VerifyScatterLoopExecution({3}, /*num_threads=*/6, /*vector_size=*/1));
}

TEST_F(CpuFusionEmitterTest, ScatterIndexingMapRejectsInvalidInputs) {
  std::unique_ptr<mlir::MLIRContext> ctx = FusionCompiler::CreateContext();
  EXPECT_DEATH(GetScatterIndexingMap({}, /*num_threads=*/1, /*vector_size=*/1,
                                     ctx.get()),
               "updates_operand_shape");
  EXPECT_DEATH(GetScatterIndexingMap({4, 4}, /*num_threads=*/0,
                                     /*vector_size=*/1, ctx.get()),
               "num_threads");
  EXPECT_DEATH(GetScatterIndexingMap({4, 4}, /*num_threads=*/1,
                                     /*vector_size=*/0, ctx.get()),
               "vector_size");
}

TEST_F(CpuFusionEmitterTest, ScatterEmptyShapesShortCircuitAndExecute) {
  constexpr absl::string_view kZeroUpdatesHlo = R"hlo(
    HloModule scatter_zero_updates
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = f32[2,2]{1,0} parameter(0)
      indices = s32[0,1]{1,0} parameter(1)
      updates = f32[0,1,2]{2,1,0} parameter(2)
      ROOT s = f32[2,2]{1,0} scatter(operand, indices, updates),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = f32[2,2]{1,0} parameter(0)
      p1 = s32[0,1]{1,0} parameter(1)
      p2 = f32[0,1,2]{2,1,0} parameter(2)
      ROOT wrapped_scatter = f32[2,2]{1,0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  constexpr absl::string_view kZeroSliceHlo = R"hlo(
    HloModule scatter_zero_slice
    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = f32[2,2]{1,0} parameter(0)
      indices = s32[2,2]{1,0} parameter(1)
      updates = f32[2,1,0]{2,1,0} parameter(2)
      ROOT s = f32[2,2]{1,0} scatter(operand, indices, updates),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0,1},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = f32[2,2]{1,0} parameter(0)
      p1 = s32[2,2]{1,0} parameter(1)
      p2 = f32[2,1,0]{2,1,0} parameter(2)
      ROOT wrapped_scatter = f32[2,2]{1,0} fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  for (absl::string_view hlo : {kZeroUpdatesHlo, kZeroSliceHlo}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                         RunBufferAssignment(*module));
    const auto* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    std::unique_ptr<mlir::MLIRContext> mlir_context =
        FusionCompiler::CreateContext();
    CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
    ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> kernel_definition,
                         emitter.EmitKernelDefinition());
    EXPECT_THAT(RunFileCheck(kernel_definition.source().ToString(), R"(
      CHECK:     func.func @wrapped_scatter(
      CHECK-NOT: xla.loop
      CHECK:     return
    )"),
                absl_testing::IsOkAndHolds(true));

    FusionCompiler compiler(mlir_context.get(),
                            FusionCompiler::Options{512, 1, true});
    ASSERT_OK_AND_ASSIGN(
        LlvmKernelSource llvm_source,
        compiler.Compile(std::move(kernel_definition).TakeSource()));
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<FunctionLibrary> library,
        JitCompileKernel(std::move(llvm_source), "wrapped_scatter"));
    ASSERT_OK_AND_ASSIGN(
        FunctionLibrary::Kernel * kernel,
        library->ResolveFunction<FunctionLibrary::Kernel>("wrapped_scatter"));

    ASSERT_OK_AND_ASSIGN(auto operand_buf, AllocateAlignedBuffer<float>(4));
    for (int i = 0; i < 4; ++i) {
      operand_buf[i] = static_cast<float>(i + 10);
    }
    ASSERT_OK_AND_ASSIGN(auto indices_buf,
                         AllocateAlignedBuffer<int32_t>(4, 99));
    std::vector<XLA_CPU_KernelArg> args = {
        {operand_buf.get(), 4 * sizeof(float)},
        {indices_buf.get(), 4 * sizeof(int32_t)},
        {nullptr, 0},
        {operand_buf.get(), 4 * sizeof(float)},
    };
    XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
    XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
    XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id,
                                     args.size(), args.data()};
    EXPECT_EQ(InvokeKernel(kernel, &frame), nullptr);
    EXPECT_THAT(absl::MakeConstSpan(operand_buf.get(), 4),
                ElementsAre(10.0f, 11.0f, 12.0f, 13.0f));
  }
}

TEST_F(CpuFusionEmitterTest, ScatterRankZeroOperandCompilesAndExecutes) {
  constexpr absl::string_view kMultiUpdateScalarHlo = R"hlo(
    HloModule scatter_rank0_multi
    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }
    scatter_computation {
      operand = s32[] parameter(0)
      indices = s32[3,0]{1,0} parameter(1)
      updates = s32[3]{0} parameter(2)
      ROOT s = s32[] scatter(operand, indices, updates),
        update_window_dims={},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={},
        index_vector_dim=1,
        to_apply=add
    }
    ENTRY main {
      p0 = s32[] parameter(0)
      p1 = s32[3,0]{1,0} parameter(1)
      p2 = s32[3]{0} parameter(2)
      ROOT wrapped_scatter = s32[] fusion(p0, p1, p2),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMultiUpdateScalarHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*module));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
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
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionLibrary> library,
      JitCompileKernel(std::move(llvm_source), "wrapped_scatter"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("wrapped_scatter"));

  ASSERT_OK_AND_ASSIGN(auto operand_buf, AllocateAlignedBuffer<int32_t>(1, 10));
  ASSERT_OK_AND_ASSIGN(auto updates_buf, AllocateAlignedBuffer<int32_t>(3));
  updates_buf[0] = 4;
  updates_buf[1] = -2;
  updates_buf[2] = 7;
  std::vector<XLA_CPU_KernelArg> args = {
      {operand_buf.get(), sizeof(int32_t)},
      {nullptr, 0},
      {updates_buf.get(), 3 * sizeof(int32_t)},
      {operand_buf.get(), sizeof(int32_t)},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ(InvokeKernel(kernel, &frame), nullptr);
  EXPECT_EQ(operand_buf[0], 19);
}

TEST_F(CpuFusionEmitterTest, VariadicScatterVectorSizeUsesWidestOperandType) {
  // Operand 0 is s8 (1 byte) and operand 1 is c128 (16 bytes). With a 64 byte
  // vectorization budget, max_vectorized_elements must be 64 / 16 = 4 (giving
  // s1 in [0, 3], s2 in [0, 3] for slice size 16), not 16 from s8 alone.
  constexpr absl::string_view kVariadicHlo = R"hlo(
    HloModule variadic_scatter_vector_size
    reducer {
      lhs0 = s8[] parameter(0)
      lhs1 = c128[] parameter(1)
      rhs0 = s8[] parameter(2)
      rhs1 = c128[] parameter(3)
      sum0 = s8[] add(lhs0, rhs0)
      sum1 = c128[] add(lhs1, rhs1)
      ROOT out = (s8[], c128[]) tuple(sum0, sum1)
    }
    scatter_computation {
      op0 = s8[4,16]{1,0} parameter(0)
      op1 = c128[4,16]{1,0} parameter(1)
      indices = s32[2,1]{1,0} parameter(2)
      upd0 = s8[2,1,16]{2,1,0} parameter(3)
      upd1 = c128[2,1,16]{2,1,0} parameter(4)
      ROOT s = (s8[4,16]{1,0}, c128[4,16]{1,0}) scatter(
          op0, op1, indices, upd0, upd1),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        to_apply=reducer
    }
    ENTRY main {
      p0 = s8[4,16]{1,0} parameter(0)
      p1 = c128[4,16]{1,0} parameter(1)
      p2 = s32[2,1]{1,0} parameter(2)
      p3 = s8[2,1,16]{2,1,0} parameter(3)
      p4 = c128[2,1,16]{2,1,0} parameter(4)
      ROOT wrapped_scatter = (s8[4,16]{1,0}, c128[4,16]{1,0}) fusion(
          p0, p1, p2, p3, p4),
        kind=kLoop, calls=scatter_computation
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kVariadicHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> buffer_assignment,
                       RunBufferAssignment(*module));
  const auto* fusion = Cast<HloFusionInstruction>(
      module->entry_computation()->root_instruction());
  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  CpuScatterFusion emitter(*buffer_assignment, fusion, mlir_context.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> kernel_definition,
                       emitter.EmitKernelDefinition());
  EXPECT_THAT(RunFileCheck(kernel_definition.source().ToString(), R"(
    CHECK: domain: thread_id in [0, 0], index_id in [0, 1], vector_id in [0, 3], vector_element_id in [0, 3]
  )"),
              absl_testing::IsOkAndHolds(true));
  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  EXPECT_THAT(compiler.Compile(std::move(kernel_definition).TakeSource()),
              absl_testing::IsOk());
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

}  // namespace
}  // namespace cpu
}  // namespace xla
