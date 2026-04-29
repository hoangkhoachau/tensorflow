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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {
namespace {

// Helper to create a simple reduction module from HLO text.
std::unique_ptr<HloModule> CreateReduceModule(int64_t n, int64_t k,
                                              bool minor) {
  std::string shape_str =
      minor ? absl::StrCat(k, ",", n) : absl::StrCat(n, ",", k);
  std::string out_shape_str = absl::StrCat(k);
  int dim_to_reduce = minor ? 1 : 0;

  std::string full_hlo = absl::StrFormat(R"(
    HloModule test_module
    add (a: f32[], b: f32[]) -> f32[] {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT result = f32[] add(a, b)
    }
    ENTRY main (param: f32[%s], init: f32[]) -> f32[%s] {
      param = f32[%s] parameter(0)
      init = f32[] parameter(1)
      ROOT result = f32[%s] reduce(param, init), dimensions={%d}, to_apply=add
    }
  )",
                                         shape_str, out_shape_str, shape_str,
                                         out_shape_str, dim_to_reduce);
  auto module_or = ParseAndReturnUnverifiedModule(full_hlo);
  CHECK_OK(module_or.status());
  return std::move(module_or).value();
}

void BM_Reduce2D(benchmark::State& state) {
  int64_t n = state.range(0);
  int64_t k = 1024;  // Fixed size for other dimension
  bool minor = state.range(1);

  auto module = CreateReduceModule(n, k, minor);
  HloEvaluator evaluator;

  Shape input_shape = minor ? ShapeUtil::MakeShape(F32, {k, n})
                            : ShapeUtil::MakeShape(F32, {n, k});
  Literal input =
      LiteralUtil::CreateFromDimensions(F32, input_shape.dimensions());
  input.PopulateWithValue(1.0f);

  Literal init = LiteralUtil::CreateR0<float>(0.0f);

  for (auto _ : state) {
    evaluator.ResetVisitStates();
    auto result =
        evaluator.Evaluate(*module->entry_computation(), {&input, &init})
            .value();
    benchmark::DoNotOptimize(result.untyped_data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_Reduce2D)
    ->Args({1024, 0})  // Major reduction
    ->Args({1024, 1})  // Minor reduction
    ->Args({1024 * 1024, 0})
    ->Args({1024 * 1024, 1});

std::unique_ptr<HloModule> CreateReduceWindowModule(int64_t n,
                                                    int64_t window_size,
                                                    int64_t stride) {
  int64_t out_size = (n - window_size) / stride + 1;
  std::string full_hlo =
      absl::StrFormat(R"(
    HloModule test_module
    add (a: f32[], b: f32[]) -> f32[] {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT result = f32[] add(a, b)
    }
    ENTRY main (param: f32[%lld], init: f32[]) -> f32[%lld] {
      param = f32[%lld] parameter(0)
      init = f32[] parameter(1)
      ROOT result = f32[%lld] reduce-window(param, init), window={size=%lld stride=%lld}, to_apply=add
    }
  )",
                      n, out_size, n, out_size, window_size, stride);
  auto module_or = ParseAndReturnUnverifiedModule(full_hlo);
  CHECK_OK(module_or.status());
  return std::move(module_or).value();
}

void BM_ReduceWindow1D(benchmark::State& state) {
  int64_t n = state.range(0);
  int64_t window_size = 128;
  int64_t stride = 64;

  auto module = CreateReduceWindowModule(n, window_size, stride);
  HloEvaluator evaluator;

  Literal input = LiteralUtil::CreateFromDimensions(F32, {n});
  input.PopulateWithValue(1.0f);

  Literal init = LiteralUtil::CreateR0<float>(0.0f);

  for (auto _ : state) {
    evaluator.ResetVisitStates();
    auto result =
        evaluator.Evaluate(*module->entry_computation(), {&input, &init})
            .value();
    benchmark::DoNotOptimize(result.untyped_data());
    benchmark::ClobberMemory();
  }
}

BENCHMARK(BM_ReduceWindow1D)->Range(1024, 1024 * 1024);

TEST(HloEvaluatorBenchmark, DummyToSatisfyTestMain) {}

}  // namespace
}  // namespace xla
