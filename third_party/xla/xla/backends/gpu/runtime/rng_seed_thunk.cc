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

#include "xla/backends/gpu/runtime/rng_seed_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/util.h"
#include "tsl/platform/random.h"

namespace xla {
namespace gpu {

uint32_t RngSeedThunk::ResolveSeed(const ExecuteParams& params) const {
  uint32_t seed = static_cast<uint32_t>(params.rng_seed);
  if (seed == 0) {
    // Generate a random non-zero seed as fallback.
    do {
      seed = static_cast<uint32_t>(tsl::random::New64());
    } while (seed == 0);
  }
  return seed;
}

absl::Status RngSeedThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto dest_addr = params.buffer_allocations->GetDeviceAddress(dest_);
  uint32_t seed = ResolveSeed(params);
  VLOG(3) << "RngSeedThunk executing with seed " << seed;
  return params.stream->Memset32(&dest_addr, seed, /*size=*/sizeof(uint32_t));
}

absl::StatusOr<const se::CommandBuffer::Command*> RngSeedThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase dst =
      execute_params.buffer_allocations->GetDeviceAddress(dest_);
  uint32_t seed = ResolveSeed(execute_params);

  VLOG(5) << "RngSeedThunk::Record: seed=" << seed;
  VLOG(5) << "  dest: " << dest_ << " (" << dst.opaque() << ")";

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateMemset(&dst, seed, /*num_elements=*/1,
                                        create->dependencies);
  }
  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(command_buffer->UpdateMemset(update->command, &dst, seed,
                                                 /*num_elements=*/1));
    return update->command;
  }
  return Internal("Invalid record action");
}

}  // namespace gpu
}  // namespace xla
