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

#include "xla/hlo/evaluator/hlo_evaluator_interpreter.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator_interpreter_ops.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/index_util.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/util.h"

namespace xla {

struct ReductionInput {
  DimensionVector dim_multipliers;
  size_t elem_size = 0;
  const Shape* shape = nullptr;  // Used if dim_multipliers is empty
  size_t index_offset = 0;
};

template <typename ConfigT, size_t InitLiteralsSize = 2>
struct GenericReduceState {
  absl::InlinedVector<ReductionInput, 2> reduction_inputs;
  ConfigT config;
  std::vector<LinearizedInterpreter::Scratchpad> scratchpads;
  absl::InlinedVector<const Literal*, InitLiteralsSize> init_literals;
  int64_t total_elements;
};

struct LinearizedInterpreter::ReduceState
    : public GenericReduceState<ReduceBatchConfig, 1> {};

struct LinearizedInterpreter::ReduceWindowState
    : public GenericReduceState<ReduceWindowBatchConfig, 2> {};

template <typename StateT>
LinearizedInterpreter::GenericReduceStateHandle<
    StateT>::~GenericReduceStateHandle() = default;

template <typename StateT>
LinearizedInterpreter::GenericReduceStateHandle<
    StateT>::GenericReduceStateHandle(std::unique_ptr<StateT> state)
    : state_(std::move(state)) {}

template <typename StateT>
LinearizedInterpreter::GenericReduceStateHandle<StateT>&
LinearizedInterpreter::GenericReduceStateHandle<StateT>::operator=(
    GenericReduceStateHandle&&) = default;

template class LinearizedInterpreter::GenericReduceStateHandle<
    LinearizedInterpreter::ReduceState>;
template class LinearizedInterpreter::GenericReduceStateHandle<
    LinearizedInterpreter::ReduceWindowState>;

namespace {

// 16 bytes is the width of SSE vector registers. We align buffers to this
// boundary to enable efficient vectorized loads and stores (SSE).
constexpr size_t kDefaultAlignment = 16;

size_t Align(size_t offset, size_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}

size_t AlignmentOfPrimitiveType(PrimitiveType type) {
  size_t alignment = 1;
  primitive_util::PrimitiveTypeSwitch<void>(
      [&](auto type_constant) {
        constexpr PrimitiveType kType = decltype(type_constant)::value;
        if constexpr (primitive_util::IsArrayType(kType)) {
          using T = primitive_util::NativeTypeOf<kType>;
          alignment = alignof(T);
        }
      },
      type);
  return alignment;
}

void IncrementIndex(const Shape& shape, absl::Span<int64_t> index) {
  if (shape.has_layout()) {
    auto minor_to_major = LayoutUtil::MinorToMajor(shape);
    for (int64_t dim : minor_to_major) {
      if (index[dim] + 1 < shape.dimensions(dim)) {
        index[dim]++;
        break;
      }
      index[dim] = 0;
    }
  } else {
    // Fallback to default major-to-minor layout (last dimension is minor).
    for (int i = shape.dimensions().size() - 1; i >= 0; --i) {
      if (index[i] + 1 < shape.dimensions(i)) {
        index[i]++;
        break;
      }
      index[i] = 0;
    }
  }
}

DimensionVector MakeDimMultipliers(const Shape& shape) {
  DimensionVector v(shape.dimensions().size());
  int64_t scale = 1;
  if (shape.has_layout()) {
    for (auto dim : LayoutUtil::MinorToMajor(shape)) {
      v[dim] = scale;
      scale *= shape.dimensions(dim);
    }
  } else {
    for (int i = shape.dimensions().size() - 1; i >= 0; --i) {
      v[i] = scale;
      scale *= shape.dimensions(i);
    }
  }
  return v;
}

template <typename F>
inline void DispatchBySize(size_t size, F&& fn) {
  switch (size) {
    case 1:
      fn(std::integral_constant<size_t, 1>{});
      break;
    case 2:
      fn(std::integral_constant<size_t, 2>{});
      break;
    case 4:
      fn(std::integral_constant<size_t, 4>{});
      break;
    case 8:
      fn(std::integral_constant<size_t, 8>{});
      break;
    case 16:
      fn(std::integral_constant<size_t, 16>{});
      break;
    default:
      fn(std::integral_constant<size_t, 0>{});
      break;  // 0 = fallback
  }
}

}  // namespace

static absl::InlinedVector<ReductionInput, 2> PrepareReductionInputs(
    absl::Span<const HloInstruction* const> input_instructions) {
  absl::InlinedVector<ReductionInput, 2> reduction_inputs(
      input_instructions.size());
  for (size_t i = 0; i < input_instructions.size(); ++i) {
    const HloInstruction* input = input_instructions[i];
    reduction_inputs[i].elem_size =
        ShapeUtil::ByteSizeOfPrimitiveType(input->shape().element_type());
    reduction_inputs[i].dim_multipliers = MakeDimMultipliers(input->shape());
    reduction_inputs[i].shape = &input->shape();
  }
  return reduction_inputs;
}

// Traces a chain of deferred operations (like Slice, Broadcast, Iota, Lookup)
// starting from 'instr'. Instead of materializing large intermediate tensors,
// these operations compute indices or values on-the-fly for a specific element
// being processed in the batch. This method records the steps needed to
// compute the final value.
absl::StatusOr<size_t> LinearizedInterpreter::TraceDeferredOpChain(
    const HloInstruction* instr, LeafLiteralResolver resolver,
    size_t input_index_offset, size_t& current_offset) {
  auto get_execute_fn =
      [&](std::optional<HloOpcode> opcode, PrimitiveType type,
          const char* op_name) -> absl::StatusOr<Step::ExecuteFn> {
    Step::ExecuteFn fn = nullptr;
    TF_RETURN_IF_ERROR(primitive_util::PrimitiveTypeSwitch<absl::Status>(
        [&](auto type_constant) -> absl::Status {
          constexpr PrimitiveType kType = decltype(type_constant)::value;
          if constexpr (kType == S32 || kType == F32 || kType == S64 ||
                        kType == BF16 || kType == F64 || kType == PRED) {
            using T = primitive_util::NativeTypeOf<kType>;
            if (opcode && *opcode == HloOpcode::kIota) {
              fn = &Ops::ExecuteIota<T>;
            } else {
              fn = &Ops::ExecuteLookup<T>;
            }
            return absl::OkStatus();
          }
          return absl::UnimplementedError(
              absl::StrCat("Unsupported ", op_name, " type"));
        },
        type));
    return fn;
  };

  if (instr->opcode() == HloOpcode::kSlice) {
    const HloInstruction* operand = instr->operand(0);
    current_offset = Align(current_offset, alignof(int64_t));
    size_t index_size =
        operand->shape().dimensions().size() * batch_size_ * sizeof(int64_t);
    size_t operand_index_offset = current_offset;
    current_offset += index_size;

    SliceData slice_data;
    const auto* slice = Cast<HloSliceInstruction>(instr);
    slice_data.starts.assign(slice->slice_starts().begin(),
                             slice->slice_starts().end());
    slice_data.strides.assign(slice->slice_strides().begin(),
                              slice->slice_strides().end());
    slice_data.rank = instr->shape().dimensions().size();

    AddStep(HloOpcode::kSlice, operand_index_offset, {input_index_offset},
            std::move(slice_data), &Ops::ExecuteSlice);

    return TraceDeferredOpChain(operand, resolver, operand_index_offset,
                                current_offset);
  }

  if (instr->opcode() == HloOpcode::kBroadcast) {
    const HloInstruction* operand = instr->operand(0);
    current_offset = Align(current_offset, alignof(int64_t));
    size_t index_size =
        operand->shape().dimensions().size() * batch_size_ * sizeof(int64_t);
    size_t operand_index_offset = current_offset;
    current_offset += index_size;

    BroadcastData broadcast_data;
    const auto* broadcast = Cast<HloBroadcastInstruction>(instr);
    broadcast_data.dimensions.assign(broadcast->dimensions().begin(),
                                     broadcast->dimensions().end());
    broadcast_data.result_rank = instr->shape().dimensions().size();
    broadcast_data.operand_rank = operand->shape().dimensions().size();

    AddStep(HloOpcode::kBroadcast, operand_index_offset, {input_index_offset},
            std::move(broadcast_data), &Ops::ExecuteBroadcast);

    return TraceDeferredOpChain(operand, resolver, operand_index_offset,
                                current_offset);
  }

  if (instr->opcode() == HloOpcode::kIota) {
    current_offset =
        Align(current_offset,
              AlignmentOfPrimitiveType(instr->shape().element_type()));
    size_t value_size =
        primitive_util::ByteWidth(instr->shape().element_type()) * batch_size_;
    size_t value_offset = current_offset;
    current_offset += value_size;

    IotaData iota_data;
    const auto* iota = Cast<HloIotaInstruction>(instr);
    iota_data.dimension = iota->iota_dimension();
    iota_data.rank = instr->shape().dimensions().size();

    PrimitiveType type = instr->shape().element_type();
    TF_ASSIGN_OR_RETURN(auto execute_fn,
                        get_execute_fn(HloOpcode::kIota, type, "Iota"));

    AddStep(HloOpcode::kIota, value_offset, {input_index_offset},
            std::move(iota_data), execute_fn, type);

    return value_offset;
  }

  // Materialized lookup
  current_offset = Align(
      current_offset, AlignmentOfPrimitiveType(instr->shape().element_type()));
  size_t value_size =
      primitive_util::ByteWidth(instr->shape().element_type()) * batch_size_;
  size_t value_offset = current_offset;
  current_offset += value_size;

  LookupData lookup_data;
  lookup_data.literal = &resolver(instr);
  lookup_data.rank = instr->shape().dimensions().size();
  lookup_data.raw_data = lookup_data.literal->untyped_data();
  lookup_data.dim_multipliers =
      MakeDimMultipliers(lookup_data.literal->shape());

  PrimitiveType type = instr->shape().element_type();
  TF_ASSIGN_OR_RETURN(auto execute_fn,
                      get_execute_fn(std::nullopt, type, "Lookup"));

  AddStep(std::nullopt, value_offset, {input_index_offset},
          std::move(lookup_data), execute_fn, type);

  return value_offset;
}

void LinearizedInterpreter::RecordInstructionOffset(const HloInstruction* instr,
                                                    size_t offset) {
  ShapeTree<size_t> offsets(instr->shape());
  *offsets.mutable_element({}) = offset;
  instruction_offsets_.emplace(instr, std::move(offsets));
}

void LinearizedInterpreter::AddStep(std::optional<HloOpcode> opcode,
                                    size_t result_offset,
                                    absl::Span<const size_t> operand_offsets,
                                    StepData data, Step::ExecuteFn execute_fn,
                                    PrimitiveType type) {
  Step step;
  step.opcode = opcode;
  step.result_offset = result_offset;
  step.operand_offsets.assign(operand_offsets.begin(), operand_offsets.end());
  step.data = std::move(data);
  step.execute_fn = execute_fn;
  step.type = type;
  step.batch_size = batch_size_;
  steps_.push_back(std::move(step));
}

absl::Status LinearizedInterpreter::PopulateStepExecuteFn(
    Step& step, const HloInstruction* instr,
    PrimitiveType promoted_type) const {
  if (step.opcode == HloOpcode::kAdd) {
    if (step.type == F32) {
      step.execute_fn = &Ops::ExecuteAdd<float, float, float>;
    } else if (step.type == F64) {
      PrimitiveType lhs_type = step.operand_types[0];
      PrimitiveType rhs_type = step.operand_types[1];
      if (lhs_type == F64 && rhs_type == F32) {
        step.execute_fn = &Ops::ExecuteAdd<double, double, float>;
      } else if (lhs_type == F32 && rhs_type == F64) {
        step.execute_fn = &Ops::ExecuteAdd<double, float, double>;
      } else if (lhs_type == F64 && rhs_type == F64) {
        step.execute_fn = &Ops::ExecuteAdd<double, double, double>;
      }
    } else if (step.type == S32) {
      step.execute_fn = &Ops::ExecuteAdd<int32_t, int32_t, int32_t>;
    }
  } else if (step.opcode == HloOpcode::kMaximum) {
    if (step.type == F32) {
      step.execute_fn = &Ops::ExecuteMaximum<float>;
    } else if (step.type == S32) {
      step.execute_fn = &Ops::ExecuteMaximum<int32_t>;
    }
  } else if (step.opcode == HloOpcode::kCompare) {
    const auto* compare = Cast<HloCompareInstruction>(instr);
    ComparisonDirection direction = compare->direction();
    PrimitiveType operand_type = step.operand_types[0];

    TF_RETURN_IF_ERROR(primitive_util::PrimitiveTypeSwitch<absl::Status>(
        [&](auto type_constant) -> absl::Status {
          constexpr PrimitiveType kType = decltype(type_constant)::value;
          if constexpr (kType == BF16 || kType == S32 || kType == S64) {
            using T = primitive_util::NativeTypeOf<kType>;
            return Ops::SetCompareExecuteFn<T>(step, direction);
          }
          return absl::UnimplementedError("Unsupported compare operand type");
        },
        operand_type));
  } else if (step.opcode == HloOpcode::kOr) {
    if (step.type == PRED) {
      step.execute_fn = &Ops::ExecuteOr;
    } else {
      return absl::UnimplementedError("Unsupported Or type");
    }
  } else if (step.opcode == HloOpcode::kAnd) {
    if (step.type == PRED) {
      step.execute_fn = &Ops::ExecuteAnd;
    } else {
      return absl::UnimplementedError("Unsupported And type");
    }
  } else if (step.opcode == HloOpcode::kSelect) {
    PrimitiveType cond_type = step.operand_types[0];
    if (cond_type != PRED) {
      return absl::UnimplementedError("Select condition must be PRED");
    }
    const HloInstruction* cond = instr->operand(0);
    if (ShapeUtil::ElementsIn(cond->shape()) !=
        ShapeUtil::ElementsIn(instr->shape())) {
      return absl::UnimplementedError(
          "Select condition must have same element count as result");
    }
    if (step.type == BF16) {
      step.execute_fn = &Ops::ExecuteSelect<bfloat16>;
    } else if (step.type == S32) {
      step.execute_fn = &Ops::ExecuteSelect<int32_t>;
    } else if (step.type == S64) {
      step.execute_fn = &Ops::ExecuteSelect<int64_t>;
    } else {
      return absl::UnimplementedError("Unsupported Select type");
    }
  } else if (step.opcode == HloOpcode::kConstant) {
    step.aux_data = instr->literal().untyped_data();
    step.execute_fn = [](const Step* s, void* base) {
      size_t bytes =
          ShapeUtil::ByteSizeOfPrimitiveType(s->type) * s->element_count;
      if (bytes > 0) {
        char* dest = static_cast<char*>(base) + s->result_offset;
        size_t src_bytes = bytes / s->batch_size;
        for (int i = 0; i < s->batch_size; ++i) {
          std::memcpy(dest + i * src_bytes, s->aux_data, src_bytes);
        }
      }
    };
  }
  return absl::OkStatus();
}

absl::Status LinearizedInterpreter::CopyBackResults(
    absl::Span<Literal> results,
    absl::Span<const std::vector<char>> temp_results, int64_t start_elem,
    int actual_batch_size, absl::Span<const int64_t> output_indices) const {
  int num_args = results.size();
  bool all_same_layout = true;
  for (int i = 1; i < num_args; ++i) {
    if (!LayoutUtil::Equal(results[i].shape().layout(),
                           results[0].shape().layout())) {
      all_same_layout = false;
      break;
    }
  }

  const int rank = results[0].shape().dimensions().size();

  for (int b = 0; b < actual_batch_size; ++b) {
    DimensionVector multi_index;
    if (!all_same_layout) {
      if (!output_indices.empty()) {
        multi_index.resize(rank);
        for (int d = 0; d < rank; ++d) {
          multi_index[d] = output_indices[b * rank + d];
        }
      } else {
        multi_index = IndexUtil::LinearIndexToMultidimensionalIndex(
            results[0].shape(), start_elem + b);
      }
    }

    for (int i = 0; i < num_args; ++i) {
      size_t elem_size =
          ShapeUtil::ByteSizeOfPrimitiveType(results[i].shape().element_type());
      int64_t linear_index;
      if (all_same_layout) {
        linear_index = start_elem + b;
      } else {
        linear_index = IndexUtil::MultidimensionalIndexToLinearIndex(
            results[i].shape(), multi_index);
      }
      char* dest = static_cast<char*>(results[i].untyped_data()) +
                   linear_index * elem_size;
      const char* src = temp_results[i].data() + b * elem_size;
      std::memcpy(dest, src, elem_size);
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<LinearizedInterpreter>>
LinearizedInterpreter::Build(const HloInstruction* instruction,
                             LeafLiteralResolver resolver, int batch_size,
                             bool precise_reduction) {
  if (batch_size <= 0) {
    return absl::InvalidArgumentError("batch_size must be > 0");
  }
  // The batch size is limited to 16 because we use a uint16_t bitmask to
  // represent valid batch lanes in EvaluateReductionBatchStep.
  if (batch_size > kMaxBatchSize) {
    return absl::InvalidArgumentError(
        absl::StrCat("batch_size must be <= ", kMaxBatchSize));
  }
  auto interpreter = absl::WrapUnique(new LinearizedInterpreter());
  interpreter->batch_size_ = batch_size;
  size_t current_offset = 0;

  const HloComputation* computation = nullptr;
  if (instruction->opcode() == HloOpcode::kReduce) {
    computation = Cast<HloReduceInstruction>(instruction)->to_apply();
  } else if (instruction->opcode() == HloOpcode::kReduceWindow) {
    computation = Cast<HloReduceWindowInstruction>(instruction)->to_apply();
  }

  absl::flat_hash_map<const HloInstruction*, size_t> operand_value_offsets;

  // Trace deferred ops first.
  auto allocate_indices = [&](const Shape& shape) -> size_t {
    current_offset = Align(current_offset, alignof(int64_t));
    size_t index_size =
        shape.dimensions().size() * batch_size * sizeof(int64_t);
    size_t index_offset = current_offset;
    current_offset += index_size;
    return index_offset;
  };

  if (computation) {
    for (const HloInstruction* operand : instruction->operands()) {
      if (operand_value_offsets.contains(operand)) {
        continue;
      }
      size_t operand_index_offset = allocate_indices(operand->shape());

      interpreter->RecordInstructionOffset(operand, operand_index_offset);

      TF_ASSIGN_OR_RETURN(
          size_t val_offset,
          interpreter->TraceDeferredOpChain(
              operand, resolver, operand_index_offset, current_offset));
      operand_value_offsets[operand] = val_offset;
    }
  } else {
    size_t index_offset = allocate_indices(instruction->shape());

    interpreter->RecordInstructionOffset(instruction, index_offset);

    interpreter->root_index_offset_ = index_offset;

    TF_ASSIGN_OR_RETURN(size_t val_offset, interpreter->TraceDeferredOpChain(
                                               instruction, resolver,
                                               index_offset, current_offset));
    interpreter->result_slots_.push_back(
        {val_offset,
         static_cast<size_t>(ShapeUtil::ByteSizeOf(instruction->shape()))});
  }

  if (computation) {
    interpreter->computation_ = computation;
    interpreter->param_is_double_.assign(computation->num_parameters(), false);

    interpreter->param_slots_.assign(computation->num_parameters(),
                                     std::nullopt);
    interpreter->param_sizes_.assign(computation->num_parameters(), 0);
    interpreter->param_elem_sizes_.assign(computation->num_parameters(), 0);

    std::vector<HloInstruction*> post_order =
        computation->MakeInstructionPostOrder();

    absl::flat_hash_map<const HloInstruction*, PrimitiveType> promoted_types;

    for (const HloInstruction* instr : post_order) {
      if (instr->opcode() == HloOpcode::kTuple) {
        ShapeTree<size_t> tuple_offsets(instr->shape());
        for (int i = 0; i < instr->operand_count(); ++i) {
          tuple_offsets.CopySubtreeFrom(
              interpreter->instruction_offsets_.at(instr->operand(i)),
              /*src_index=*/{},
              /*dst_index=*/{i});
        }
        interpreter->instruction_offsets_.emplace(instr,
                                                  std::move(tuple_offsets));
        continue;
      }

      if (instr->opcode() == HloOpcode::kGetTupleElement) {
        const ShapeTree<size_t>& operand_offsets =
            interpreter->instruction_offsets_.at(instr->operand(0));
        ShapeTree<size_t> gte_offsets(instr->shape());
        gte_offsets.CopySubtreeFrom(operand_offsets,
                                    /*src_index=*/{instr->tuple_index()},
                                    /*dst_index=*/{});
        interpreter->instruction_offsets_.emplace(instr,
                                                  std::move(gte_offsets));
        continue;
      }

      PrimitiveType promoted_type = instr->shape().element_type();
      if (precise_reduction && promoted_type == F32) {
        if (instr->opcode() == HloOpcode::kParameter) {
          if (instr->parameter_number() < computation->num_parameters() / 2) {
            promoted_type = F64;
            interpreter->param_is_double_[instr->parameter_number()] = true;
          }
        } else if (instr->opcode() == HloOpcode::kAdd) {
          if (promoted_types[instr->operand(0)] == F64 ||
              promoted_types[instr->operand(1)] == F64) {
            promoted_type = F64;
          }
        }
      }
      promoted_types[instr] = promoted_type;

      // Allocate space for normal array instructions.
      current_offset =
          Align(current_offset, AlignmentOfPrimitiveType(promoted_type));
      size_t size_bytes = primitive_util::ByteWidth(promoted_type) *
                          ShapeUtil::ElementsIn(instr->shape()) * batch_size;

      if (instr->shape().IsTuple()) {
        VLOG(1) << "Cannot linearize: Tuples are only supported for kTuple and "
                   "kGetTupleElement. Found: "
                << instr->ToString();
        return absl::UnimplementedError(
            "Tuples are only supported for kTuple and kGetTupleElement");
      }

      interpreter->RecordInstructionOffset(instr, current_offset);

      if (instr->opcode() == HloOpcode::kParameter) {
        int param_no = instr->parameter_number();
        int num_args = computation->num_parameters() / 2;
        if (param_no >= num_args) {
          int operand_idx = param_no - num_args;
          const HloInstruction* operand = instruction->operand(operand_idx);

          if (operand_value_offsets.contains(operand)) {
            size_t val_offset = operand_value_offsets[operand];

            interpreter->param_slots_[param_no] = val_offset;
            interpreter->param_sizes_[param_no] = size_bytes;
            interpreter->param_elem_sizes_[param_no] =
                primitive_util::ByteWidth(promoted_type);

            *interpreter->instruction_offsets_.at(instr).mutable_element({}) =
                val_offset;

            continue;
          }
        }

        interpreter->param_slots_[param_no] = current_offset;
        interpreter->param_sizes_[param_no] = size_bytes;
        interpreter->param_elem_sizes_[param_no] =
            primitive_util::ByteWidth(promoted_type);
        current_offset += size_bytes;
        continue;
      }

      Step step;
      step.opcode = instr->opcode();
      step.type = promoted_type;
      step.result_offset = current_offset;
      step.element_count = ShapeUtil::ElementsIn(instr->shape()) * batch_size;
      step.batch_size = batch_size;

      for (const HloInstruction* operand : instr->operands()) {
        step.operand_offsets.push_back(
            interpreter->instruction_offsets_.at(operand).element({}));
        step.operand_types.push_back(promoted_types[operand]);
      }

      TF_RETURN_IF_ERROR(
          interpreter->PopulateStepExecuteFn(step, instr, promoted_type));

      if (step.execute_fn) {
        interpreter->steps_.push_back(step);
      } else {
        VLOG(1) << "Cannot linearize: Unsupported op in interpreter: "
                << HloOpcodeString(instr->opcode()) << " for type "
                << primitive_util::LowercasePrimitiveTypeName(step.type);
        return absl::UnimplementedError(
            absl::StrCat("Unsupported op in interpreter: ",
                         HloOpcodeString(instr->opcode())));
      }

      current_offset += size_bytes;
    }

    for (int64_t i = 0; i < computation->num_parameters(); ++i) {
      if (!interpreter->param_slots_[i].has_value()) {
        const HloInstruction* param = computation->parameter_instruction(i);
        size_t size_bytes = ShapeUtil::ByteSizeOf(param->shape()) * batch_size;
        current_offset =
            Align(current_offset,
                  AlignmentOfPrimitiveType(param->shape().element_type()));
        interpreter->param_slots_[i] = current_offset;
        interpreter->param_sizes_[i] = size_bytes;
        interpreter->param_elem_sizes_[i] =
            ShapeUtil::ByteSizeOf(param->shape());
        current_offset += size_bytes;
      }
    }

    const HloInstruction* root = computation->root_instruction();
    const ShapeTree<size_t>& root_offsets =
        interpreter->instruction_offsets_.at(root);
    root_offsets.ForEachElement([&](const ShapeIndex& index, size_t offset) {
      if (ShapeUtil::IsLeafIndex(root->shape(), index)) {
        const Shape& sub_shape = ShapeUtil::GetSubshape(root->shape(), index);
        size_t size_bytes = ShapeUtil::ByteSizeOf(sub_shape);
        interpreter->result_slots_.push_back({offset, size_bytes});
      }
    });
  }

  interpreter->scratchpad_size_ = Align(current_offset, kDefaultAlignment);
  return std::move(interpreter);
}

LinearizedInterpreter::ReduceBatchConfig
LinearizedInterpreter::CreateReduceBatchConfig(
    const Shape& input_shape, absl::Span<const int64_t> dimensions_to_reduce,
    absl::Span<const DimensionVector> input_dim_multipliers) {
  ReduceBatchConfig config;

  absl::Span<const int64_t> arg_dimensions = input_shape.dimensions();
  std::vector<int64_t> arg_dim_steps(arg_dimensions.size(), 0);
  for (const int64_t dim : dimensions_to_reduce) {
    arg_dim_steps[dim] = 1;
  }

  std::vector<int64_t> result_to_arg_index;
  for (int64_t i = 0; i < arg_dimensions.size(); ++i) {
    if (arg_dim_steps[i] == 0) {
      result_to_arg_index.push_back(i);
    }
  }

  std::vector<int64_t> arg_to_result_index(arg_dimensions.size(), -1);
  for (size_t d = 0; d < result_to_arg_index.size(); ++d) {
    arg_to_result_index[result_to_arg_index[d]] = d;
  }

  std::vector<int64_t> layout_reduced_dims;
  for (int64_t dim : LayoutUtil::MinorToMajor(input_shape)) {
    if (absl::c_linear_search(dimensions_to_reduce, dim)) {
      layout_reduced_dims.push_back(dim);
    }
  }
  absl::c_reverse(layout_reduced_dims);

  std::vector<int64_t> reduced_dims;
  reduced_dims.reserve(layout_reduced_dims.size());
  for (const int64_t dim : layout_reduced_dims) {
    reduced_dims.push_back(input_shape.dimensions(dim));
  }
  Shape reduced_shape =
      ShapeUtil::MakeShape(input_shape.element_type(), reduced_dims);

  config.reduced_shape = reduced_shape;
  config.layout_reduced_dims = layout_reduced_dims;
  config.result_to_arg_index = result_to_arg_index;
  config.arg_to_result_index = arg_to_result_index;
  config.arg_dim_steps = arg_dim_steps;
  config.input_dim_multipliers.assign(input_dim_multipliers.begin(),
                                      input_dim_multipliers.end());

  return config;
}

LinearizedInterpreter::ReduceWindowBatchConfig
LinearizedInterpreter::CreateReduceWindowBatchConfig(const Shape& input_shape,
                                                     const Window& window) {
  ReduceWindowBatchConfig config;

  absl::InlinedVector<int64_t, 2> window_dimension_sizes;
  for (const auto& window_dimension : window.dimensions()) {
    window_dimension_sizes.push_back(window_dimension.size());
  }
  const Shape window_shape =
      ShapeUtil::MakeShape(input_shape.element_type(), window_dimension_sizes);

  const size_t rank = window_shape.dimensions().size();
  absl::InlinedVector<int64_t, 8> input_dimensions(rank);
  for (size_t d = 0; d < rank; ++d) {
    input_dimensions[d] = input_shape.dimensions(d);
  }

  absl::InlinedVector<LinearizedInterpreter::WindowDim, 8> cached_window_dims(
      rank);
  for (size_t d = 0; d < rank; ++d) {
    const auto& dim = window.dimensions(d);
    cached_window_dims[d] = {dim.stride(), dim.window_dilation(),
                             dim.padding_low(), dim.base_dilation()};
  }

  config.window_shape = window_shape;
  config.window_dims.assign(cached_window_dims.begin(),
                            cached_window_dims.end());
  config.input_dimensions.assign(input_dimensions.begin(),
                                 input_dimensions.end());

  return config;
}

template <typename StateT, typename ConfigBuilderFn>
std::unique_ptr<StateT> LinearizedInterpreter::CreateGenericReduceStateInternal(
    const HloInstruction* instruction,
    absl::Span<const HloInstruction* const> inputs,
    absl::Span<const Literal* const> init_literals, int num_threads,
    ConfigBuilderFn&& config_builder) const {
  auto state = std::make_unique<StateT>();
  state->init_literals.assign(init_literals.begin(), init_literals.end());

  const Shape& output_shape = instruction->shape().IsTuple()
                                  ? instruction->shape().tuple_shapes(0)
                                  : instruction->shape();
  state->total_elements = ShapeUtil::ElementsIn(output_shape);

  state->reduction_inputs = PrepareReductionInputs(inputs);

  for (size_t i = 0; i < inputs.size(); ++i) {
    const HloInstruction* input = inputs[i];
    CHECK(instruction_offsets_.contains(input))
        << "Reduction input not found in instruction_offsets_: "
        << input->name();
    state->reduction_inputs[i].index_offset =
        instruction_offsets_.at(input).element({});
  }

  state->config = config_builder(inputs, state->reduction_inputs);

  int count = num_threads + 1;
  state->scratchpads.reserve(count);
  for (int i = 0; i < count; ++i) {
    state->scratchpads.push_back(CreateScratchpad());
  }

  return state;
}

LinearizedInterpreter::ReduceStateHandle
LinearizedInterpreter::CreateReduceState(
    const HloInstruction* reduce_instruction,
    absl::Span<const Literal* const> init_literals, int num_threads) const {
  CHECK(computation_ != nullptr)
      << "CreateReduceState called on a non-reduction interpreter";
  const auto* reduce = Cast<HloReduceInstruction>(reduce_instruction);
  auto config_builder = [&](auto inputs, const auto& reduction_inputs) {
    std::vector<DimensionVector> dim_multipliers(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      dim_multipliers[i] = reduction_inputs[i].dim_multipliers;
    }
    return CreateReduceBatchConfig(inputs[0]->shape(), reduce->dimensions(),
                                   absl::MakeSpan(dim_multipliers));
  };
  auto state = CreateGenericReduceStateInternal<ReduceState>(
      reduce_instruction, reduce->inputs(), init_literals, num_threads,
      config_builder);
  return ReduceStateHandle(std::move(state));
}

LinearizedInterpreter::ReduceWindowStateHandle
LinearizedInterpreter::CreateReduceWindowState(
    const HloInstruction* reduce_window_instruction,
    absl::Span<const Literal* const> init_literals, int num_threads) const {
  CHECK(computation_ != nullptr)
      << "CreateReduceWindowState called on a non-reduction interpreter";
  const auto* reduce_window =
      Cast<HloReduceWindowInstruction>(reduce_window_instruction);
  auto config_builder = [&](auto inputs, const auto& reduction_inputs) {
    return CreateReduceWindowBatchConfig(inputs[0]->shape(),
                                         reduce_window->window());
  };
  auto state = CreateGenericReduceStateInternal<ReduceWindowState>(
      reduce_window_instruction, reduce_window->inputs(), init_literals,
      num_threads, config_builder);
  return ReduceWindowStateHandle(std::move(state));
}

LinearizedInterpreter::Scratchpad LinearizedInterpreter::CreateScratchpad()
    const {
  Scratchpad s(scratchpad_size_);
  char* base = static_cast<char*>(s.data());
  for (const auto& step : steps_) {
    if (step.opcode == HloOpcode::kConstant) {
      step.execute_fn(&step, base);
    }
  }
  return s;
}

void LinearizedInterpreter::ExecuteSteps(Scratchpad& scratchpad) const {
  char* base = static_cast<char*>(scratchpad.data());
  for (const Step& step : steps_) {
    if (step.opcode != HloOpcode::kConstant) {
      step.execute_fn(&step, base);
    }
  }
}

absl::Status LinearizedInterpreter::InitializeAccumulators(
    Scratchpad& scratchpad, absl::Span<const void* const> init_values) const {
  int num_args = init_values.size();
  char* base = static_cast<char*>(scratchpad.data());
  for (int i = 0; i < num_args; ++i) {
    if (!param_slots_[i].has_value()) {
      return absl::InternalError("Accumulator parameter slot not found.");
    }
    size_t offset = param_slots_[i].value();
    size_t elem_size = param_elem_sizes_[i];
    if (elem_size == 0) {
      continue;
    }
    char* dest_base = base + offset;
    const void* init_val_ptr = init_values[i];

    if (param_is_double_[i]) {
      // If precise_reduction is enabled, we promote F32 accumulators to F64.
      // However, the initial values passed in are still the original F32
      // values. Therefore, we must read them as float and convert to double
      // before storing them in the F64 accumulator slots. We know it is a float
      // because param_is_double_ is only set to true when the original
      // parameter type was F32.
      float val_float;
      std::memcpy(&val_float, init_val_ptr, sizeof(float));
      double val_double = static_cast<double>(val_float);
      for (int b = 0; b < batch_size_; ++b) {
        std::memcpy(dest_base + b * 8, &val_double, 8);
      }
      continue;
    }

    DispatchBySize(elem_size, [&](auto size_tag) {
      constexpr size_t N = decltype(size_tag)::value;
      if constexpr (N > 0) {
        char val[N];
        std::memcpy(val, init_val_ptr, N);
        for (int b = 0; b < batch_size_; ++b) {
          std::memcpy(dest_base + b * N, val, N);
        }
      } else {
        for (int b = 0; b < batch_size_; ++b) {
          std::memcpy(dest_base + b * elem_size, init_val_ptr, elem_size);
        }
      }
    });
  }
  return absl::OkStatus();
}

absl::Status LinearizedInterpreter::EvaluateReductionBatchStep(
    Scratchpad& scratchpad, uint16_t out_of_bounds_mask) const {
  int num_args = result_slots_.size();
  char* base = static_cast<char*>(scratchpad.data());

  // 2. Save accumulators for out-of-bounds lanes if mask is provided
  auto& saved_accumulators = scratchpad.saved_accumulators;
  if (out_of_bounds_mask != 0) {
    saved_accumulators.resize(num_args);
    for (int i = 0; i < num_args; ++i) {
      size_t elem_size = param_elem_sizes_[i];
      if (elem_size == 0) {
        continue;
      }
      saved_accumulators[i].resize(batch_size_ * elem_size);
      // Bulk copy instead of masked copy
      std::memcpy(saved_accumulators[i].data(), base + param_slots_[i].value(),
                  batch_size_ * elem_size);
    }
  }

  // 3. Execute steps
  ExecuteSteps(scratchpad);

  // 4. Ping-Pong results to accumulators
  for (int i = 0; i < num_args; ++i) {
    const auto& slot = result_slots_[i];
    if (param_slots_[i].value() != slot.offset) {
      size_t total_size = param_elem_sizes_[i] * batch_size_;
      std::memcpy(base + param_slots_[i].value(), base + slot.offset,
                  total_size);
    }
  }

  // 5. Restore accumulators for out-of-bounds lanes in param_slots_ (since
  // Ping-Pong just overwrote them)
  if (out_of_bounds_mask != 0) {
    for (int i = 0; i < num_args; ++i) {
      size_t elem_size = param_elem_sizes_[i];
      if (elem_size == 0) {
        continue;
      }
      char* dest_base = base + param_slots_[i].value();
      const char* src_base = saved_accumulators[i].data();

      DispatchBySize(elem_size, [&](auto size_tag) {
        constexpr size_t N = decltype(size_tag)::value;
        if constexpr (N > 0) {
          for (int b = 0; b < batch_size_; ++b) {
            if ((out_of_bounds_mask >> b) & 1) {
              std::memcpy(dest_base + b * N, src_base + b * N, N);
            }
          }
        } else {
          for (int b = 0; b < batch_size_; ++b) {
            if ((out_of_bounds_mask >> b) & 1) {
              std::memcpy(dest_base + b * elem_size, src_base + b * elem_size,
                          elem_size);
            }
          }
        }
      });
    }
  }

  return absl::OkStatus();
}

absl::Status LinearizedInterpreter::ExtractAccumulators(
    Scratchpad& scratchpad, absl::Span<void* const> results) const {
  int num_args = results.size();
  char* base = static_cast<char*>(scratchpad.data());
  for (int i = 0; i < num_args; ++i) {
    if (!param_slots_[i].has_value()) {
      return absl::InternalError("Accumulator parameter slot not found.");
    }
    size_t offset = param_slots_[i].value();
    size_t total_size = param_sizes_[i];

    if (param_is_double_[i]) {
      // Convert double to float
      char* src_base = base + offset;
      char* dest_base = static_cast<char*>(results[i]);
      for (int b = 0; b < batch_size_; ++b) {
        double val_double;
        std::memcpy(&val_double, src_base + b * 8, 8);
        float val_float = static_cast<float>(val_double);
        std::memcpy(dest_base + b * sizeof(float), &val_float, sizeof(float));
      }
      continue;
    }

    std::memcpy(results[i], base + offset, total_size);
  }
  return absl::OkStatus();
}

absl::Status LinearizedInterpreter::EvaluateDeferredOpBatch(
    Scratchpad& scratchpad, absl::Span<const int64_t> linear_indices,
    const Shape& shape, void* output_buffer) const {
  if (computation_ != nullptr) {
    return absl::InternalError(
        "EvaluateDeferredOpBatch called on a reduction interpreter");
  }
  if (linear_indices.empty()) {
    return absl::OkStatus();
  }
  if (linear_indices.size() > batch_size_) {
    return absl::InvalidArgumentError(
        absl::StrCat("linear_indices size (", linear_indices.size(),
                     ") exceeds batch size (", batch_size_, ")"));
  }
  int rank = shape.dimensions().size();
  char* base = static_cast<char*>(scratchpad.data());

  for (int b = 0; b < linear_indices.size(); ++b) {
    auto multi_index =
        IndexUtil::LinearIndexToMultidimensionalIndex(shape, linear_indices[b]);
    char* dest = base + root_index_offset_ + b * rank * sizeof(int64_t);
    std::memcpy(dest, multi_index.data(), rank * sizeof(int64_t));
  }

  for (int b = linear_indices.size(); b < batch_size_; ++b) {
    char* dest = base + root_index_offset_ + b * rank * sizeof(int64_t);
    std::memset(dest, 0, rank * sizeof(int64_t));
  }

  ExecuteSteps(scratchpad);

  if (result_slots_.empty()) {
    return absl::InternalError(
        "No result slot found for deferred op materialization.");
  }
  size_t result_offset = result_slots_[0].offset;
  size_t elem_size = ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  char* src = base + result_offset;

  std::memcpy(output_buffer, src, linear_indices.size() * elem_size);

  return absl::OkStatus();
}

absl::Status LinearizedInterpreter::EvaluateReduceBatch(
    ReduceStateHandle& state_handle, absl::Span<Literal> results,
    int64_t batch_idx, int thread_id) const {
  ReduceState& state = *state_handle;
  int64_t start_elem = batch_idx * batch_size_;
  int actual_batch_size =
      std::min<int64_t>(batch_size_, state.total_elements - start_elem);
  if (actual_batch_size <= 0) {
    return absl::OkStatus();
  }
  int num_args = state.reduction_inputs.size();

  absl::InlinedVector<const void*, 2> init_ptrs(num_args);
  for (int i = 0; i < num_args; ++i) {
    init_ptrs[i] = state.init_literals[i]->untyped_data();
  }

  // Reusable buffers from scratchpad
  auto& scratchpad = state.scratchpads[thread_id + 1];
  auto& temp_results = scratchpad.temp_results;

  temp_results.resize(num_args);

  TF_RETURN_IF_ERROR(InitializeAccumulators(scratchpad, init_ptrs));

  absl::InlinedVector<ReductionInput, 2> batch_reduction_inputs(num_args);
  for (int i = 0; i < num_args; ++i) {
    batch_reduction_inputs[i] = state.reduction_inputs[i];
  }

  // Pre-populate scratchpad with invariant parts of output indices.
  auto output_index = IndexUtil::LinearIndexToMultidimensionalIndex(
      results[0].shape(), start_elem);
  for (int b = 0; b < batch_size_; ++b) {
    if (b < actual_batch_size) {
      for (int i = 0; i < num_args; ++i) {
        int arg_rank = state.reduction_inputs[i].shape->dimensions().size();
        int64_t* dest_indices = scratchpad.GetIndicesPointer(
            state.reduction_inputs[i].index_offset, b, arg_rank);
        // Initialize with zeros as default for reduced dimensions.
        std::fill(dest_indices, dest_indices + arg_rank, 0);
        for (int64_t j = 0; j < output_index.size(); ++j) {
          dest_indices[state.config.result_to_arg_index[j]] = output_index[j];
        }
      }
      if (b + 1 < actual_batch_size) {
        IncrementIndex(results[0].shape(), absl::MakeSpan(output_index));
      }
    } else {
      // Zero-initialize indices for inactive lanes to avoid OOB read in
      // ExecuteLookup.
      for (int i = 0; i < num_args; ++i) {
        int arg_rank = state.reduction_inputs[i].shape->dimensions().size();
        int64_t* dest_indices = scratchpad.GetIndicesPointer(
            state.reduction_inputs[i].index_offset, b, arg_rank);
        std::fill(dest_indices, dest_indices + arg_rank, 0);
      }
    }
  }

  std::vector<int64_t> reduced_index(state.config.layout_reduced_dims.size(),
                                     0);

  const int rank = state.config.layout_reduced_dims.size();

  if (ShapeUtil::ElementsIn(state.config.reduced_shape) > 0) {
    uint16_t valid_batch_mask = ((1U << actual_batch_size) - 1);
    uint16_t out_of_bounds_mask = static_cast<uint16_t>(~valid_batch_mask);
    do {
      for (int i = 0; i < num_args; ++i) {
        int arg_rank = state.reduction_inputs[i].shape->dimensions().size();
        for (int b = 0; b < actual_batch_size; ++b) {
          int64_t* dest_indices = scratchpad.GetIndicesPointer(
              state.reduction_inputs[i].index_offset, b, arg_rank);
          for (size_t k = 0; k < rank; ++k) {
            dest_indices[state.config.layout_reduced_dims[k]] =
                reduced_index[k];
          }
        }
      }

      TF_RETURN_IF_ERROR(
          EvaluateReductionBatchStep(scratchpad, out_of_bounds_mask));
    } while (IndexUtil::BumpIndices(state.config.reduced_shape,
                                    absl::MakeSpan(reduced_index)));
  }

  absl::InlinedVector<void*, 2> results_ptrs(num_args);
  for (int i = 0; i < num_args; ++i) {
    size_t elem_size =
        ShapeUtil::ByteSizeOfPrimitiveType(results[i].shape().element_type());
    temp_results[i].resize(batch_size_ * elem_size);
    results_ptrs[i] = temp_results[i].data();
  }

  TF_RETURN_IF_ERROR(ExtractAccumulators(scratchpad, results_ptrs));

  return CopyBackResults(results, temp_results, start_elem, actual_batch_size);
}

absl::Status LinearizedInterpreter::EvaluateReduceWindowBatch(
    ReduceWindowStateHandle& state_handle, absl::Span<Literal> results,
    int64_t batch_idx, int thread_id) const {
  ReduceWindowState& state = *state_handle;
  int64_t start_elem = batch_idx * batch_size_;
  int actual_batch_size =
      std::min<int64_t>(batch_size_, state.total_elements - start_elem);
  if (actual_batch_size <= 0) {
    return absl::OkStatus();
  }
  int num_args = state.reduction_inputs.size();

  absl::InlinedVector<const void*, 2> init_ptrs(num_args);
  for (int i = 0; i < num_args; ++i) {
    init_ptrs[i] = state.init_literals[i]->untyped_data();
  }

  int rank = results[0].shape().dimensions().size();

  // Reusable buffers from scratchpad
  auto& scratchpad = state.scratchpads[thread_id + 1];
  auto& base_index_invariants = scratchpad.base_index_invariants;
  auto& base_indices = scratchpad.base_indices;
  auto& output_indices = scratchpad.output_indices;
  auto& temp_results = scratchpad.temp_results;

  base_index_invariants.resize(batch_size_ * rank);
  base_indices.resize(batch_size_ * rank);
  output_indices.resize(batch_size_ * rank);
  temp_results.resize(num_args);

  absl::InlinedVector<ReductionInput, 2> batch_reduction_inputs(num_args);
  for (int i = 0; i < num_args; ++i) {
    batch_reduction_inputs[i] = state.reduction_inputs[i];
  }

  TF_RETURN_IF_ERROR(InitializeAccumulators(scratchpad, init_ptrs));

  // Precompute base linear indices for this batch
  auto output_index = IndexUtil::LinearIndexToMultidimensionalIndex(
      results[0].shape(), start_elem);
  for (int b = 0; b < actual_batch_size; ++b) {
    for (int d = 0; d < rank; ++d) {
      output_indices[b * rank + d] = output_index[d];
    }
    if (b + 1 < actual_batch_size) {
      IncrementIndex(results[0].shape(), absl::MakeSpan(output_index));
    }
  }

  for (size_t d = 0; d < rank; ++d) {
    int64_t stride = state.config.window_dims[d].stride;
    int64_t padding_low = state.config.window_dims[d].padding_low;
    for (int b = 0; b < actual_batch_size; ++b) {
      base_index_invariants[d * batch_size_ + b] =
          output_indices[b * rank + d] * stride - padding_low;
    }
  }

  bool is_interior = true;
  for (size_t d = 0; d < rank; ++d) {
    if (state.config.window_dims[d].base_dilation != 1) {
      is_interior = false;
      break;
    }
    int64_t min_window_offset = 0;
    int64_t max_window_offset = (state.config.window_shape.dimensions(d) - 1) *
                                state.config.window_dims[d].window_dilation;
    uint64_t input_dim_u =
        static_cast<uint64_t>(state.config.input_dimensions[d]);
    for (int b = 0; b < actual_batch_size; ++b) {
      int64_t invariant = base_index_invariants[d * batch_size_ + b];
      if (invariant + min_window_offset < 0 ||
          static_cast<uint64_t>(invariant + max_window_offset) >= input_dim_u) {
        is_interior = false;
        break;
      }
    }
    if (!is_interior) {
      break;
    }
  }
  uint16_t valid_batch_mask = ((1U << actual_batch_size) - 1);

  DimensionVector window_index(rank, 0);
  absl::Span<int64_t> window_index_span = absl::MakeSpan(window_index);

  int64_t* base_indices_ptr = base_indices.data();
  const int64_t* invariants_ptr = base_index_invariants.data();

  absl::InlinedVector<int64_t, 16> window_offsets(rank);

  if (ShapeUtil::ElementsIn(state.config.window_shape) > 0) {
    do {
      // is_padding_mask is a bitmask where bit 'b' is set if the window element
      // for batch lane 'b' falls outside the valid input bounds (i.e., in the
      // padding). This mask is passed to EvaluateReductionBatchStep to ensure
      // that out-of-bounds lanes do not participate in the reduction or
      // overwrite valid accumulators.
      uint16_t is_padding_mask = 0;

      for (size_t d = 0; d < rank; ++d) {
        window_offsets[d] =
            window_index_span[d] * state.config.window_dims[d].window_dilation;
      }

      if (is_interior) {
        for (size_t d = 0; d < rank; ++d) {
          const int64_t window_offset = window_offsets[d];
          for (int b = 0; b < actual_batch_size; ++b) {
            base_indices_ptr[d * batch_size_ + b] =
                invariants_ptr[d * batch_size_ + b] + window_offset;
          }
        }
      } else {
        for (size_t d = 0; d < rank; ++d) {
          const int64_t window_offset = window_offsets[d];
          const int64_t base_dilation =
              state.config.window_dims[d].base_dilation;
          const uint64_t input_dim_u =
              static_cast<uint64_t>(state.config.input_dimensions[d]);

          if (base_dilation != 1) {
            for (int b = 0; b < actual_batch_size; ++b) {
              int64_t base_index_i =
                  invariants_ptr[d * batch_size_ + b] + window_offset;
              bool out = (base_index_i % base_dilation != 0);
              base_index_i /= base_dilation;
              base_indices_ptr[d * batch_size_ + b] = base_index_i;
              out = out || (static_cast<uint64_t>(base_index_i) >= input_dim_u);
              if (out) {
                is_padding_mask |= (1U << b);
              }
            }
          } else {
            for (int b = 0; b < actual_batch_size; ++b) {
              int64_t base_index_i =
                  invariants_ptr[d * batch_size_ + b] + window_offset;
              base_indices_ptr[d * batch_size_ + b] = base_index_i;
              bool out = static_cast<uint64_t>(base_index_i) >= input_dim_u;
              if (out) {
                is_padding_mask |= (1U << b);
              }
            }
          }
        }
      }

      if (!is_interior &&
          (is_padding_mask & valid_batch_mask) == valid_batch_mask) {
        continue;
      }

      for (int i = 0; i < num_args; ++i) {
        int arg_rank = state.reduction_inputs[i].shape->dimensions().size();
        for (int b = 0; b < batch_size_; ++b) {
          int64_t* dest_indices = scratchpad.GetIndicesPointer(
              state.reduction_inputs[i].index_offset, b, arg_rank);

          if (b < actual_batch_size) {
            bool is_padding = (is_padding_mask >> b) & 1;
            if (!is_padding) {
              for (int d = 0; d < arg_rank; ++d) {
                dest_indices[d] = base_indices_ptr[d * batch_size_ + b];
              }
            } else {
              std::fill(dest_indices, dest_indices + arg_rank, 0);
            }
          } else {
            // Zero-initialize indices for inactive lanes to avoid OOB read in
            // ExecuteLookup.
            std::fill(dest_indices, dest_indices + arg_rank, 0);
          }
        }
      }

      TF_RETURN_IF_ERROR(EvaluateReductionBatchStep(
          scratchpad,
          static_cast<uint16_t>(is_padding_mask | ~valid_batch_mask)));
    } while (
        IndexUtil::BumpIndices(state.config.window_shape, window_index_span));
  }

  absl::InlinedVector<void*, 2> results_ptrs(num_args);
  for (int i = 0; i < num_args; ++i) {
    size_t elem_size =
        ShapeUtil::ByteSizeOfPrimitiveType(results[i].shape().element_type());
    temp_results[i].resize(batch_size_ * elem_size);
    results_ptrs[i] = temp_results[i].data();
  }

  TF_RETURN_IF_ERROR(ExtractAccumulators(scratchpad, results_ptrs));

  return CopyBackResults(results, temp_results, start_elem, actual_batch_size,
                         absl::MakeSpan(output_indices));
}

}  // namespace xla
