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

#ifndef XLA_BACKENDS_GPU_RUNTIME_RNG_SEED_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_RNG_SEED_THUNK_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class RngSeedThunk : public Command {
 public:
  RngSeedThunk(ThunkInfo thunk_info, const BufferAllocation::Slice& dest)
      : Command(CommandType::kRngSeedCmd, Kind::kRngSeed,
                std::move(thunk_info)),
        dest_(dest) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferAllocation::Slice dest() const { return dest_; }

  BufferUses buffer_uses() const override {
    return {BufferUse::Write(dest_, ShapeUtil::MakeShape(U32, {}))};
  }

 private:
  // Generate random seed if requested, otherwise use current params seed.
  uint32_t ResolveSeed(const ExecuteParams& params) const;

  const BufferAllocation::Slice dest_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_RNG_SEED_THUNK_H_
