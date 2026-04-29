/* Copyright 2026 The OpenXLA Authors.

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
#ifndef XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_H_
#define XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape_tree.h"
#include "xla/xla_data.pb.h"

namespace xla {

// LinearizedInterpreter evaluates a subset of HLO instructions by linearizing
// the computation graph into a sequence of flat steps. This approach avoids the
// overhead of recursive evaluation and repeated index delinearization
// (converting linear indices to multi-dimensional indices and back) by
// processing elements in a flat buffer. It also supports batching (up to
// kMaxBatchSize) to process multiple elements simultaneously, which is
// particularly useful for improving efficiency in reductions and element-wise
// operations.
class LinearizedInterpreter {
 public:
  using LeafLiteralResolver =
      absl::FunctionRef<const Literal&(const HloInstruction*)>;

  // Scratchpad is the unified working memory for the interpreter. It holds
  // operand pointers, result pointers, saved accumulators for reduction edge
  // cases, and temporary indices. By consolidating all intermediate buffers
  // into a single allocation (or a few vectors), it minimizes heap allocations
  // during interpretation.
  class Scratchpad {
   public:
    explicit Scratchpad(size_t size) : buffer_(size) {}
    void* data() { return buffer_.data(); }

    int64_t* GetIndicesPointer(size_t offset, int batch_idx, int rank) {
      return reinterpret_cast<int64_t*>(static_cast<char*>(data()) + offset +
                                        batch_idx * rank * sizeof(int64_t));
    }

   private:
    friend class LinearizedInterpreter;

    std::vector<const void*> args_ptrs;
    std::vector<void*> results_ptrs;
    std::vector<std::vector<char>> saved_accumulators;

    std::vector<std::vector<int64_t>> batched_linear_indices;
    std::vector<std::vector<char>> batch_deferred_buffers;
    std::vector<int64_t> base_index_invariants;
    std::vector<int64_t> base_indices;
    std::vector<int64_t> output_indices;
    std::vector<std::vector<char>> temp_results;

    std::vector<char> buffer_;
  };

  static constexpr int kMaxBatchSize = 16;

  // batch_size must be <= kMaxBatchSize.
  static absl::StatusOr<std::unique_ptr<LinearizedInterpreter>> Build(
      const HloInstruction* instruction, LeafLiteralResolver resolver,
      int batch_size = 1, bool precise_reduction = false);

  // ReduceState and ReduceWindowState hold the configuration and execution
  // state for reduction and reduce-window operations, respectively. They manage
  // state across multiple threads and batches.
  struct ReduceState;
  struct ReduceWindowState;

  template <typename StateT>
  class GenericReduceStateHandle {
   public:
    ~GenericReduceStateHandle();
    GenericReduceStateHandle(GenericReduceStateHandle&&) = default;
    GenericReduceStateHandle& operator=(GenericReduceStateHandle&&);

    StateT& operator*() { return *state_; }
    StateT* operator->() { return state_.get(); }
    const StateT& operator*() const { return *state_; }
    const StateT* operator->() const { return state_.get(); }

   private:
    friend class LinearizedInterpreter;
    explicit GenericReduceStateHandle(std::unique_ptr<StateT> state);
    std::unique_ptr<StateT> state_;
  };

  using ReduceStateHandle = GenericReduceStateHandle<ReduceState>;
  using ReduceWindowStateHandle = GenericReduceStateHandle<ReduceWindowState>;

  ReduceStateHandle CreateReduceState(
      const HloInstruction* reduce_instruction,
      absl::Span<const Literal* const> init_literals, int num_threads) const;

  ReduceWindowStateHandle CreateReduceWindowState(
      const HloInstruction* reduce_window_instruction,
      absl::Span<const Literal* const> init_literals, int num_threads) const;

  Scratchpad CreateScratchpad() const;

  // Extracts final accumulators from scratchpad to results.
  absl::Status ExtractAccumulators(Scratchpad& scratchpad,
                                   absl::Span<void* const> results) const;

  absl::Status EvaluateDeferredOpBatch(Scratchpad& scratchpad,
                                       absl::Span<const int64_t> linear_indices,
                                       const Shape& shape,
                                       void* output_buffer) const;

  absl::Status EvaluateReduceBatch(ReduceStateHandle& state,
                                   absl::Span<Literal> results,
                                   int64_t batch_idx, int thread_id) const;

  absl::Status EvaluateReduceWindowBatch(ReduceWindowStateHandle& state,
                                         absl::Span<Literal> results,
                                         int64_t batch_idx,
                                         int thread_id) const;

 private:
  LinearizedInterpreter() = default;

  class Ops;

  struct IotaData {
    int64_t dimension;
    int rank;
  };

  struct BroadcastData {
    DimensionVector dimensions;
    int result_rank;
    int operand_rank;
  };

  struct SliceData {
    DimensionVector starts;
    DimensionVector strides;
    int rank;
  };

  struct LookupData {
    const Literal* literal;
    int rank;
    const void* raw_data = nullptr;
    DimensionVector dim_multipliers;
  };

  using StepData = std::variant<std::monostate, IotaData, BroadcastData,
                                SliceData, LookupData>;

  struct Step {
    using ExecuteFn = void (*)(const Step*, void* /*scratchpad_base*/);

    ExecuteFn execute_fn = nullptr;
    std::optional<HloOpcode> opcode;
    PrimitiveType type;

    size_t result_offset;
    std::vector<size_t> operand_offsets;
    std::vector<PrimitiveType> operand_types;
    size_t element_count = 0;
    const void* aux_data = nullptr;
    int batch_size = 1;
    StepData data;
  };

  absl::Status PopulateStepExecuteFn(Step& step, const HloInstruction* instr,
                                     PrimitiveType promoted_type) const;

  absl::Status CopyBackResults(
      absl::Span<Literal> results,
      absl::Span<const std::vector<char>> temp_results, int64_t start_elem,
      int actual_batch_size,
      absl::Span<const int64_t> output_indices = {}) const;

  void AddStep(std::optional<HloOpcode> opcode, size_t result_offset,
               absl::Span<const size_t> operand_offsets, StepData data,
               Step::ExecuteFn execute_fn,
               PrimitiveType type = PRIMITIVE_TYPE_INVALID);

  void RecordInstructionOffset(const HloInstruction* instr, size_t offset);

  template <typename StateT, typename ConfigBuilderFn>
  std::unique_ptr<StateT> CreateGenericReduceStateInternal(
      const HloInstruction* instruction,
      absl::Span<const HloInstruction* const> inputs,
      absl::Span<const Literal* const> init_literals, int num_threads,
      ConfigBuilderFn&& config_creator) const;

  absl::StatusOr<size_t> TraceDeferredOpChain(const HloInstruction* instr,
                                              LeafLiteralResolver resolver,
                                              size_t input_index_offset,
                                              size_t& current_offset);

  // Executes the scheduled steps on the scratchpad.
  void ExecuteSteps(Scratchpad& scratchpad) const;

  // Initializes accumulators in scratchpad with init_values.
  absl::Status InitializeAccumulators(
      Scratchpad& scratchpad, absl::Span<const void* const> init_values) const;

  // Evaluates a batch of reduction steps.
  absl::Status EvaluateReductionBatchStep(
      Scratchpad& scratchpad, uint16_t out_of_bounds_mask = 0) const;

  struct ReduceBatchConfig {
    Shape reduced_shape;
    std::vector<int64_t> layout_reduced_dims;
    std::vector<int64_t> result_to_arg_index;
    std::vector<int64_t> arg_to_result_index;
    std::vector<int64_t> arg_dim_steps;
    std::vector<DimensionVector> input_dim_multipliers;
  };

  struct WindowDim {
    int64_t stride;
    int64_t window_dilation;
    int64_t padding_low;
    int64_t base_dilation;
  };

  struct ReduceWindowBatchConfig {
    Shape window_shape;
    std::vector<WindowDim> window_dims;
    std::vector<int64_t> input_dimensions;
  };

  static ReduceBatchConfig CreateReduceBatchConfig(
      const Shape& input_shape, absl::Span<const int64_t> dimensions_to_reduce,
      absl::Span<const DimensionVector> input_dim_multipliers);

  static ReduceWindowBatchConfig CreateReduceWindowBatchConfig(
      const Shape& input_shape, const Window& window);

  std::vector<Step> steps_;
  size_t scratchpad_size_ = 0;
  size_t root_index_offset_ = 0;
  std::vector<bool> param_is_double_;

  // Mapping from instruction to its allocated offsets in the scratchpad.
  absl::flat_hash_map<const HloInstruction*, ShapeTree<size_t>>
      instruction_offsets_;

  // Slots for parameters and results to easily map args/results in Evaluate.
  // These store the byte offsets in the scratchpad.
  std::vector<std::optional<size_t>> param_slots_;
  std::vector<size_t> param_sizes_;
  std::vector<size_t> param_elem_sizes_;

  struct ResultSlot {
    size_t offset;
    size_t size;
  };
  std::vector<ResultSlot> result_slots_;

  const HloComputation* computation_ = nullptr;
  int batch_size_ = 1;
};

}  // namespace xla

#endif  // XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_H_
