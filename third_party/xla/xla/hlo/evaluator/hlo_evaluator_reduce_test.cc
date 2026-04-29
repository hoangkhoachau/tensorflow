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
#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/shape_inference.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

class HloEvaluatorReduceTest : public HloHardwareIndependentTestBase {
 protected:
  std::unique_ptr<HloModule> m_ = CreateNewVerifiedModule();

  void ComparePaths(HloComputation* computation,
                    absl::Span<const Literal* const> args) {
    HloEvaluator evaluator_fast;
    evaluator_fast.set_reduce_use_fast_path(true);

    HloEvaluator evaluator_slow;
    evaluator_slow.set_reduce_use_fast_path(false);

    auto fast_result_or = evaluator_fast.Evaluate(*computation, args);
    ASSERT_TRUE(fast_result_or.ok()) << fast_result_or.status().message();
    auto fast_result = std::move(fast_result_or).value();

    auto slow_result_or = evaluator_slow.Evaluate(*computation, args);
    ASSERT_TRUE(slow_result_or.ok()) << slow_result_or.status().message();
    auto slow_result = std::move(slow_result_or).value();

    std::function<void(Literal, Literal)> compare = [&](Literal slow,
                                                        Literal fast) {
      if (slow.shape().IsTuple()) {
        EXPECT_TRUE(fast.shape().IsTuple());
        if (!fast.shape().IsTuple()) return;
        auto slow_decomposed = std::move(slow).DecomposeTuple();
        auto fast_decomposed = std::move(fast).DecomposeTuple();
        EXPECT_EQ(slow_decomposed.size(), fast_decomposed.size());
        if (slow_decomposed.size() != fast_decomposed.size()) return;
        for (size_t i = 0; i < slow_decomposed.size(); ++i) {
          compare(std::move(slow_decomposed[i]), std::move(fast_decomposed[i]));
        }
      } else {
        if (ShapeUtil::ElementIsFloating(slow.shape())) {
          EXPECT_TRUE(LiteralTestUtil::Near(slow, fast, ErrorSpec(1e-4, 1e-4)));
        } else {
          EXPECT_TRUE(LiteralTestUtil::Equal(slow, fast));
        }
      }
    };

    compare(std::move(slow_result), std::move(fast_result));
  }

  HloComputation* CreateAddComputation(PrimitiveType type) {
    HloComputation::Builder b("add");
    Shape scalar_shape = ShapeUtil::MakeShape(type, {});
    auto param_lhs = b.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "lhs"));
    auto param_rhs = b.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "rhs"));
    b.AddInstruction(HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd,
                                                  param_lhs, param_rhs));
    return m_->AddEmbeddedComputation(b.Build());
  }

  HloComputation* CreateVariadicAddComputation() {
    HloComputation::Builder b("variadic_add");
    Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
    auto p0 = b.AddInstruction(
        HloInstruction::CreateParameter(0, scalar_shape, "acc0"));
    auto p1 = b.AddInstruction(
        HloInstruction::CreateParameter(1, scalar_shape, "acc1"));
    auto p2 = b.AddInstruction(
        HloInstruction::CreateParameter(2, scalar_shape, "val0"));
    auto p3 = b.AddInstruction(
        HloInstruction::CreateParameter(3, scalar_shape, "val1"));

    auto add0 = b.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p0, p2));
    auto add1 = b.AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, p1, p3));

    b.AddInstruction(HloInstruction::CreateTuple({add0, add1}));
    return m_->AddEmbeddedComputation(b.Build());
  }
};

TEST_F(HloEvaluatorReduceTest, SimpleReduce) {
  HloComputation::Builder b(TestName());
  auto input = b.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4}), "input"));
  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto add_func = CreateAddComputation(F32);
  b.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(F32, {}), input, init_value, {0}, add_func));

  m_->AddEntryComputation(b.Build());
  Literal arg = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  ComparePaths(m_->entry_computation(), {&arg});
}

TEST_F(HloEvaluatorReduceTest, ReduceWindowWithDilation) {
  HloComputation::Builder b(TestName());
  auto input = b.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4, 4}), "input"));
  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto add_func = CreateAddComputation(F32);

  Window window;
  WindowDimension* dim0 = window.add_dimensions();
  dim0->set_size(2);
  dim0->set_stride(1);
  dim0->set_padding_low(0);
  dim0->set_padding_high(0);
  dim0->set_window_dilation(2);
  dim0->set_base_dilation(1);

  WindowDimension* dim1 = window.add_dimensions();
  dim1->set_size(2);
  dim1->set_stride(1);
  dim1->set_padding_low(0);
  dim1->set_padding_high(0);
  dim1->set_window_dilation(1);
  dim1->set_base_dilation(2);

  Shape output_shape = ShapeUtil::MakeShape(F32, {2, 6});

  b.AddInstruction(HloInstruction::CreateReduceWindow(
      output_shape, input, init_value, window, add_func));

  m_->AddEntryComputation(b.Build());

  Literal arg(ShapeUtil::MakeShape(F32, {4, 4}));
  int counter = 0;
  ASSERT_TRUE(arg.Populate<float>([&](absl::Span<const int64_t>) {
                   return static_cast<float>(++counter);
                 })
                  .ok());

  ComparePaths(m_->entry_computation(), {&arg});
}

TEST_F(HloEvaluatorReduceTest, ReduceWithNonDefaultLayout) {
  HloComputation::Builder b(TestName());
  Shape input_shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {4, 4}, {0, 1});  // Minor-to-major: 0, 1
  auto input = b.AddInstruction(
      HloInstruction::CreateParameter(0, input_shape, "input"));
  auto init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto add_func = CreateAddComputation(F32);
  b.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(F32, {4}), input, init_value, {0}, add_func));

  m_->AddEntryComputation(b.Build());

  Literal arg(input_shape);
  int counter = 0;
  ASSERT_TRUE(arg.Populate<float>([&](absl::Span<const int64_t>) {
                   return static_cast<float>(++counter);
                 })
                  .ok());

  ComparePaths(m_->entry_computation(), {&arg});
}

TEST_F(HloEvaluatorReduceTest, VariadicReduce) {
  HloComputation::Builder b(TestName());
  auto input0 = b.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4}), "input0"));
  auto input1 = b.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(F32, {4}), "input1"));
  auto init_value0 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto init_value1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));

  auto add_func = CreateVariadicAddComputation();

  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})});

  b.AddInstruction(HloInstruction::CreateReduce(tuple_shape, {input0, input1},
                                                {init_value0, init_value1}, {0},
                                                add_func));

  m_->AddEntryComputation(b.Build());

  Literal arg0 = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Literal arg1 = LiteralUtil::CreateR1<float>({5.0f, 6.0f, 7.0f, 8.0f});
  ComparePaths(m_->entry_computation(), {&arg0, &arg1});
}

TEST_F(HloEvaluatorReduceTest, VariadicReduceDuplicateOperands) {
  HloComputation::Builder b(TestName());
  auto input = b.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {4, 4}), "input"));
  auto init_value0 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto init_value1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));

  auto add_func = CreateVariadicAddComputation();

  Shape tuple_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})});

  b.AddInstruction(HloInstruction::CreateReduce(tuple_shape, {input, input},
                                                {init_value0, init_value1},
                                                {0, 1}, add_func));

  m_->AddEntryComputation(b.Build());

  Literal arg(ShapeUtil::MakeShape(F32, {4, 4}));
  arg.PopulateWithValue<float>(1.0f);

  ComparePaths(m_->entry_computation(), {&arg});

  HloEvaluator evaluator;
  auto result_or = evaluator.Evaluate(*m_->entry_computation(), {&arg});
  ASSERT_TRUE(result_or.ok());
  Literal result = std::move(result_or).value();

  ASSERT_TRUE(result.shape().IsTuple());
  auto decomposed = std::move(result).DecomposeTuple();
  ASSERT_EQ(decomposed.size(), 2);
  EXPECT_FLOAT_EQ(decomposed[0].Get<float>({}), 16.0f);
  EXPECT_FLOAT_EQ(decomposed[1].Get<float>({}), 16.0f);
}

struct ReduceTestCase {
  PrimitiveType type;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> reduce_dims;
  bool use_iota;
};

class HloEvaluatorReduceParamTest
    : public HloEvaluatorReduceTest,
      public ::testing::WithParamInterface<ReduceTestCase> {};

TEST_P(HloEvaluatorReduceParamTest, ComparePaths) {
  const auto& param = GetParam();
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(param.type, param.input_shape);
  HloInstruction* input;
  if (param.use_iota) {
    input = b.AddInstruction(HloInstruction::CreateIota(shape, 0));
  } else {
    input =
        b.AddInstruction(HloInstruction::CreateParameter(0, shape, "input"));
  }

  HloInstruction* init_value;
  if (param.type == F32) {
    init_value = b.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  } else {
    init_value = b.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  }

  auto add_func = CreateAddComputation(param.type);

  std::vector<int64_t> output_dims;
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    if (!absl::c_linear_search(param.reduce_dims, i)) {
      output_dims.push_back(shape.dimensions(i));
    }
  }
  Shape output_shape = ShapeUtil::MakeShape(param.type, output_dims);

  b.AddInstruction(HloInstruction::CreateReduce(output_shape, input, init_value,
                                                param.reduce_dims, add_func));

  m_->AddEntryComputation(b.Build());

  if (param.use_iota) {
    ComparePaths(m_->entry_computation(), {});
  } else {
    Literal arg(shape);
    int counter = 0;
    if (param.type == F32) {
      ASSERT_TRUE(arg.Populate<float>([&](absl::Span<const int64_t>) {
                       return static_cast<float>(++counter);
                     })
                      .ok());
    } else {
      ASSERT_TRUE(arg.Populate<int32_t>(
                         [&](absl::Span<const int64_t>) { return ++counter; })
                      .ok());
    }
    ComparePaths(m_->entry_computation(), {&arg});
  }
}

INSTANTIATE_TEST_SUITE_P(
    HloEvaluatorReduceParamTest_Instances, HloEvaluatorReduceParamTest,
    ::testing::Values(ReduceTestCase{F32, {4}, {0}, false},
                      ReduceTestCase{F32, {4, 4}, {0}, false},
                      ReduceTestCase{F32, {4, 4}, {1}, false},
                      ReduceTestCase{F32, {4, 4}, {0, 1}, false},
                      ReduceTestCase{F32, {4, 4, 4}, {1}, false},
                      ReduceTestCase{F32, {4, 4, 4}, {0, 2}, false},
                      ReduceTestCase{F32, {4}, {0}, true},
                      ReduceTestCase{F32, {4, 4}, {0}, true},
                      ReduceTestCase{F32, {4, 4}, {1}, true},
                      // S32 cases
                      ReduceTestCase{S32, {4}, {0}, false},
                      ReduceTestCase{S32, {4, 4}, {0}, false},
                      ReduceTestCase{S32, {4, 4}, {1}, false},
                      // Zero-sized dims
                      ReduceTestCase{F32, {0}, {0}, false},
                      ReduceTestCase{F32, {4, 0}, {0}, false},
                      ReduceTestCase{F32, {4, 0}, {1}, false}));

struct ReduceWindowTestCase {
  PrimitiveType type;
  std::vector<int64_t> input_shape;
  std::vector<int64_t> window_size;
  std::vector<int64_t> window_stride;
  std::vector<int64_t> padding_low;
  std::vector<int64_t> padding_high;
  std::vector<int64_t> window_dilation;
  std::vector<int64_t> base_dilation;
  bool use_iota;
};

class HloEvaluatorReduceWindowParamTest
    : public HloEvaluatorReduceTest,
      public ::testing::WithParamInterface<ReduceWindowTestCase> {};

TEST_P(HloEvaluatorReduceWindowParamTest, ComparePaths) {
  const auto& param = GetParam();
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(param.type, param.input_shape);
  HloInstruction* input;
  if (param.use_iota) {
    input = b.AddInstruction(HloInstruction::CreateIota(shape, 0));
  } else {
    input =
        b.AddInstruction(HloInstruction::CreateParameter(0, shape, "input"));
  }

  HloInstruction* init_value;
  if (param.type == F32) {
    init_value = b.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  } else {
    init_value = b.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  }

  auto add_func = CreateAddComputation(param.type);

  Window window;
  for (size_t i = 0; i < param.input_shape.size(); ++i) {
    WindowDimension* dim = window.add_dimensions();
    dim->set_size(param.window_size[i]);
    dim->set_stride(param.window_stride[i]);
    dim->set_padding_low(param.padding_low[i]);
    dim->set_padding_high(param.padding_high[i]);
    dim->set_window_dilation(param.window_dilation[i]);
    dim->set_base_dilation(param.base_dilation[i]);
  }

  std::vector<int64_t> output_dims;
  for (size_t i = 0; i < param.input_shape.size(); ++i) {
    int64_t in_dim = param.input_shape[i];
    int64_t w_size = param.window_size[i];
    int64_t stride = param.window_stride[i];
    int64_t pad_low = param.padding_low[i];
    int64_t pad_high = param.padding_high[i];
    int64_t w_dil = param.window_dilation[i];
    int64_t b_dil = param.base_dilation[i];

    int64_t dilated_in = (in_dim - 1) * b_dil + 1;
    int64_t padded_in = dilated_in + pad_low + pad_high;
    int64_t eff_w_size = (w_size - 1) * w_dil + 1;
    CHECK_GE(padded_in, eff_w_size) << "Window is larger than padded input";
    int64_t out_dim = (padded_in - eff_w_size) / stride + 1;
    output_dims.push_back(out_dim);
  }
  Shape output_shape = ShapeUtil::MakeShape(param.type, output_dims);

  b.AddInstruction(HloInstruction::CreateReduceWindow(
      output_shape, input, init_value, window, add_func));

  m_->AddEntryComputation(b.Build());

  if (param.use_iota) {
    ComparePaths(m_->entry_computation(), {});
  } else {
    Literal arg(shape);
    int counter = 0;
    if (param.type == F32) {
      ASSERT_TRUE(arg.Populate<float>([&](absl::Span<const int64_t>) {
                       return static_cast<float>(++counter);
                     })
                      .ok());
    } else {
      ASSERT_TRUE(arg.Populate<int32_t>(
                         [&](absl::Span<const int64_t>) { return ++counter; })
                      .ok());
    }
    ComparePaths(m_->entry_computation(), {&arg});
  }
}

INSTANTIATE_TEST_SUITE_P(
    HloEvaluatorReduceWindowParamTest_Instances,
    HloEvaluatorReduceWindowParamTest,
    ::testing::Values(
        // 1D
        ReduceWindowTestCase{F32, {4}, {2}, {1}, {0}, {0}, {1}, {1}, false},
        ReduceWindowTestCase{F32, {4}, {2}, {2}, {0}, {0}, {1}, {1}, false},
        // Dilation
        ReduceWindowTestCase{F32, {4}, {2}, {1}, {0}, {0}, {2}, {1}, false},
        ReduceWindowTestCase{F32, {4}, {2}, {1}, {0}, {0}, {1}, {2}, false},
        // 2D
        ReduceWindowTestCase{
            F32, {4, 4}, {2, 2}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, false},
        // Complex case (window and base dilation)
        ReduceWindowTestCase{
            F32, {4, 4}, {2, 2}, {1, 1}, {0, 0}, {0, 0}, {2, 1}, {1, 2}, false},
        // Padding cases
        ReduceWindowTestCase{F32, {4}, {2}, {1}, {1}, {1}, {1}, {1}, false},
        ReduceWindowTestCase{
            F32, {4, 4}, {2, 2}, {1, 1}, {1, 0}, {0, 1}, {1, 1}, {1, 1}, false},
        // Iota cases
        ReduceWindowTestCase{F32, {4}, {2}, {1}, {0}, {0}, {1}, {1}, true},
        ReduceWindowTestCase{
            F32, {4, 4}, {2, 2}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, true},
        // S32 cases
        ReduceWindowTestCase{S32, {4}, {2}, {1}, {0}, {0}, {1}, {1}, false},
        ReduceWindowTestCase{S32,
                             {4, 4},
                             {2, 2},
                             {1, 1},
                             {0, 0},
                             {0, 0},
                             {1, 1},
                             {1, 1},
                             false}));

TEST_F(HloEvaluatorReduceTest, DeepChainDeferredOps) {
  HloComputation::Builder b(TestName());

  Shape leaf_shape = ShapeUtil::MakeShape(F32, {10, 10});

  auto* iota = b.AddInstruction(HloInstruction::CreateIota(leaf_shape, 0));
  Shape slice_shape = ShapeUtil::MakeShape(F32, {5, 5});
  auto* slice_iota = b.AddInstruction(HloInstruction::CreateSlice(
      slice_shape, iota, /*start_indices=*/{2, 2}, /*limit_indices=*/{7, 7},
      /*strides=*/{1, 1}));
  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {2, 5, 5});
  auto* broadcast_iota = b.AddInstruction(
      HloInstruction::CreateBroadcast(broadcast_shape, slice_iota, {1, 2}));

  auto add_func = CreateAddComputation(F32);
  Shape reduce_shape = ShapeUtil::MakeShape(F32, {5});
  auto* init_value = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));

  b.AddInstruction(
      HloInstruction::CreateReduce(reduce_shape, broadcast_iota, init_value,
                                   /*dimensions_to_reduce=*/{0, 2}, add_func));

  m_->AddEntryComputation(b.Build());

  ComparePaths(m_->entry_computation(), {});
}

}  // namespace
}  // namespace xla
