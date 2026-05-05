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

#include "xla/backends/cpu/runtime/rng_seed_thunk.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/random.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<RngSeedThunk>> RngSeedThunk::Create(
    Info info, BufferAllocation::Slice dest_buffer) {
  return absl::WrapUnique(new RngSeedThunk(std::move(info), dest_buffer));
}

RngSeedThunk::RngSeedThunk(Info info, BufferAllocation::Slice dest_buffer)
    : Thunk(Kind::kRngSeed, std::move(info)), dest_buffer_(dest_buffer) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> RngSeedThunk::Execute(
    const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(
      se::DeviceAddressBase dest_data,
      params.buffer_allocations->GetDeviceAddress(dest_buffer_));

  if (dest_data.size() != sizeof(uint32_t)) {
    return InvalidArgument("Invalid seed buffer size: %d", dest_data.size());
  }

  uint32_t seed = static_cast<uint32_t>(params.rng_seed);
  if (seed == 0) {
    do {
      seed = static_cast<uint32_t>(tsl::random::New64());
    } while (seed == 0);
  }

  VLOG(3) << absl::StreamFormat("RngSeedThunk: seed=%d", seed);
  VLOG(3) << absl::StreamFormat("  dest: %s (%p)", dest_buffer_.ToString(),
                                dest_data.opaque());

  std::memcpy(dest_data.opaque(), &seed, sizeof(uint32_t));

  return OkExecuteEvent();
}

Thunk::BufferUses RngSeedThunk::buffer_uses() const {
  return {BufferUse::Write(dest_buffer_, ShapeUtil::MakeShape(U32, {}))};
}

}  // namespace xla::cpu
