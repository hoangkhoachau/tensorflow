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

#ifndef XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_OPS_H_
#define XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_OPS_H_

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <variant>

#include "absl/status/status.h"
#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator_interpreter.h"

namespace xla {

class LinearizedInterpreter::Ops {
 public:
  template <typename ResT, typename LhsT, typename RhsT>
  static void ExecuteAdd(const Step* step, void* scratchpad_base) {
    ResT* result = reinterpret_cast<ResT*>(static_cast<char*>(scratchpad_base) +
                                           step->result_offset);
    const LhsT* lhs = reinterpret_cast<const LhsT*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[0]);
    const RhsT* rhs = reinterpret_cast<const RhsT*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[1]);

    for (size_t i = 0; i < step->element_count; ++i) {
      result[i] = static_cast<ResT>(lhs[i]) + static_cast<ResT>(rhs[i]);
    }
  }

  template <typename T>
  static void ExecuteMaximum(const Step* step, void* scratchpad_base) {
    T* result = reinterpret_cast<T*>(static_cast<char*>(scratchpad_base) +
                                     step->result_offset);
    const T* lhs = reinterpret_cast<const T*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[0]);
    const T* rhs = reinterpret_cast<const T*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[1]);

    for (size_t i = 0; i < step->element_count; ++i) {
      if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(lhs[i])) {
          result[i] = lhs[i];
        } else if (std::isnan(rhs[i])) {
          result[i] = rhs[i];
        } else {
          result[i] = std::max(lhs[i], rhs[i]);
        }
      } else {
        result[i] = std::max(lhs[i], rhs[i]);
      }
    }
  }

  template <typename T, ComparisonDirection Direction>
  static void ExecuteCompare(const Step* step, void* scratchpad_base) {
    static_assert(Direction == ComparisonDirection::kGt ||
                      Direction == ComparisonDirection::kNe ||
                      Direction == ComparisonDirection::kEq ||
                      Direction == ComparisonDirection::kLt ||
                      Direction == ComparisonDirection::kGe ||
                      Direction == ComparisonDirection::kLe,
                  "Unsupported compare direction");

    bool* result = reinterpret_cast<bool*>(static_cast<char*>(scratchpad_base) +
                                           step->result_offset);
    const T* lhs = reinterpret_cast<const T*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[0]);
    const T* rhs = reinterpret_cast<const T*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[1]);

    for (size_t i = 0; i < step->element_count; ++i) {
      if constexpr (Direction == ComparisonDirection::kGt) {
        result[i] = lhs[i] > rhs[i];
      } else if constexpr (Direction == ComparisonDirection::kNe) {
        result[i] = lhs[i] != rhs[i];
      } else if constexpr (Direction == ComparisonDirection::kEq) {
        result[i] = lhs[i] == rhs[i];
      } else if constexpr (Direction == ComparisonDirection::kLt) {
        result[i] = lhs[i] < rhs[i];
      } else if constexpr (Direction == ComparisonDirection::kGe) {
        result[i] = lhs[i] >= rhs[i];
      } else if constexpr (Direction == ComparisonDirection::kLe) {
        result[i] = lhs[i] <= rhs[i];
      }
    }
  }

  static void ExecuteOr(const Step* step, void* scratchpad_base) {
    bool* result = reinterpret_cast<bool*>(static_cast<char*>(scratchpad_base) +
                                           step->result_offset);
    const bool* lhs = reinterpret_cast<const bool*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[0]);
    const bool* rhs = reinterpret_cast<const bool*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[1]);

    for (size_t i = 0; i < step->element_count; ++i) {
      result[i] = lhs[i] || rhs[i];
    }
  }

  static void ExecuteAnd(const Step* step, void* scratchpad_base) {
    bool* result = reinterpret_cast<bool*>(static_cast<char*>(scratchpad_base) +
                                           step->result_offset);
    const bool* lhs = reinterpret_cast<const bool*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[0]);
    const bool* rhs = reinterpret_cast<const bool*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[1]);

    for (size_t i = 0; i < step->element_count; ++i) {
      result[i] = lhs[i] && rhs[i];
    }
  }

  template <typename T>
  static void ExecuteSelect(const Step* step, void* scratchpad_base) {
    T* result = reinterpret_cast<T*>(static_cast<char*>(scratchpad_base) +
                                     step->result_offset);
    const bool* cond = reinterpret_cast<const bool*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[0]);
    const T* lhs = reinterpret_cast<const T*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[1]);
    const T* rhs = reinterpret_cast<const T*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[2]);

    for (size_t i = 0; i < step->element_count; ++i) {
      result[i] = cond[i] ? lhs[i] : rhs[i];
    }
  }

  static void ExecuteSlice(const Step* step, void* scratchpad_base) {
    const auto& data = std::get<SliceData>(step->data);
    const int64_t* input_indices = reinterpret_cast<const int64_t*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[0]);
    int64_t* output_indices = reinterpret_cast<int64_t*>(
        static_cast<char*>(scratchpad_base) + step->result_offset);

    int rank = data.rank;
    for (int b = 0; b < step->batch_size; ++b) {
      for (int d = 0; d < rank; ++d) {
        output_indices[b * rank + d] =
            data.starts[d] + input_indices[b * rank + d] * data.strides[d];
      }
    }
  }

  static void ExecuteBroadcast(const Step* step, void* scratchpad_base) {
    const auto& data = std::get<BroadcastData>(step->data);
    const int64_t* input_indices = reinterpret_cast<const int64_t*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[0]);
    int64_t* output_indices = reinterpret_cast<int64_t*>(
        static_cast<char*>(scratchpad_base) + step->result_offset);

    int operand_rank = data.operand_rank;
    int result_rank = data.result_rank;
    for (int b = 0; b < step->batch_size; ++b) {
      for (int d = 0; d < operand_rank; ++d) {
        int mapped_dim = data.dimensions[d];
        output_indices[b * operand_rank + d] =
            input_indices[b * result_rank + mapped_dim];
      }
    }
  }

  template <typename T>
  static void ExecuteIota(const Step* step, void* scratchpad_base) {
    const auto& data = std::get<IotaData>(step->data);
    const int64_t* indices = reinterpret_cast<const int64_t*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[0]);
    T* result = reinterpret_cast<T*>(static_cast<char*>(scratchpad_base) +
                                     step->result_offset);

    int rank = data.rank;
    for (int b = 0; b < step->batch_size; ++b) {
      int64_t idx = indices[b * rank + data.dimension];
      result[b] = static_cast<T>(idx);
    }
  }

  template <typename T>
  static void ExecuteLookup(const Step* step, void* scratchpad_base) {
    const auto& data = std::get<LookupData>(step->data);
    const int64_t* indices = reinterpret_cast<const int64_t*>(
        static_cast<char*>(scratchpad_base) + step->operand_offsets[0]);
    T* result = reinterpret_cast<T*>(static_cast<char*>(scratchpad_base) +
                                     step->result_offset);

    int rank = data.rank;
    const T* raw_data = reinterpret_cast<const T*>(data.raw_data);

    int b = 0;
    // Unroll the loop by a factor of 4 to reduce loop overhead and improve
    // instruction-level parallelism. This calculates linear indices and fetches
    // data for 4 batch lanes in parallel before falling back to a scalar
    // cleanup loop.
    for (; b <= step->batch_size - 4; b += 4) {
      int64_t linear_index0 = 0;
      int64_t linear_index1 = 0;
      int64_t linear_index2 = 0;
      int64_t linear_index3 = 0;
      for (int d = 0; d < rank; ++d) {
        int64_t m = data.dim_multipliers[d];
        linear_index0 += indices[b * rank + d] * m;
        linear_index1 += indices[(b + 1) * rank + d] * m;
        linear_index2 += indices[(b + 2) * rank + d] * m;
        linear_index3 += indices[(b + 3) * rank + d] * m;
      }
      result[b] = raw_data[linear_index0];
      result[b + 1] = raw_data[linear_index1];
      result[b + 2] = raw_data[linear_index2];
      result[b + 3] = raw_data[linear_index3];
    }

    // Cleanup loop for remaining elements
    for (; b < step->batch_size; ++b) {
      int64_t linear_index = 0;
      for (int d = 0; d < rank; ++d) {
        int64_t m = data.dim_multipliers[d];
        linear_index += indices[b * rank + d] * m;
      }
      result[b] = raw_data[linear_index];
    }
  }

  template <typename T>
  static absl::Status SetCompareExecuteFn(Step& step,
                                          ComparisonDirection direction) {
    switch (direction) {
      case ComparisonDirection::kGt:
        step.execute_fn = &ExecuteCompare<T, ComparisonDirection::kGt>;
        break;
      case ComparisonDirection::kNe:
        step.execute_fn = &ExecuteCompare<T, ComparisonDirection::kNe>;
        break;
      case ComparisonDirection::kEq:
        step.execute_fn = &ExecuteCompare<T, ComparisonDirection::kEq>;
        break;
      case ComparisonDirection::kLt:
        step.execute_fn = &ExecuteCompare<T, ComparisonDirection::kLt>;
        break;
      case ComparisonDirection::kGe:
        step.execute_fn = &ExecuteCompare<T, ComparisonDirection::kGe>;
        break;
      case ComparisonDirection::kLe:
        step.execute_fn = &ExecuteCompare<T, ComparisonDirection::kLe>;
        break;
      default:
        return absl::UnimplementedError("Unsupported compare direction");
    }
    return absl::OkStatus();
  }
};

}  // namespace xla

#endif  // XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_OPS_H_
