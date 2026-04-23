// RUN: sdy_opt %s -split-input-file -xla-sdy-stablehlo-export-pipeline='enable-reduce-scatter-export=true' 2>&1 | FileCheck %s

sdy.mesh @mesh_2 = <["x"=8, "y"=4]>

// Verify the SPMD boundary custom calls (FullToShard and ShardToFull) are created around the outlined manual call,
// and that the input mhlo.copy is marked with unreduced sharding to prevent gSPMD from inserting extra reductions.
// CHECK-LABEL: func @reduce_scatter
func.func @reduce_scatter(%arg0: tensor<8x8xf32> {sdy.sharding=#sdy.sharding<@mesh_2, [{"x"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[COPY:.*]] = mhlo.copy %arg0 {mhlo.sharding = "{devices=[8,1,4]<=[32] last_tile_dims={unreduced}}"} : tensor<8x8xf32>
  // CHECK-NEXT: %[[FULL_TO_SHARD:.*]] = stablehlo.custom_call @SPMDFullToShardShape(%[[COPY]]) {mhlo.sharding = "{manual}"} : (tensor<8x8xf32>) -> tensor<1x8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @{{xla\.sdy\.inlinable_manual_computation_body(_[0-9]+)?}}(%[[FULL_TO_SHARD]]) {mhlo.sharding = "{manual}"} : (tensor<1x8xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: %[[COPY_BACK:.*]] = mhlo.copy %[[CALL]] {mhlo.sharding = "{manual}"}
  // CHECK-NEXT: %[[SHARD_TO_FULL:.*]] = stablehlo.custom_call @SPMDShardToFullShape(%[[COPY_BACK]]) {mhlo.sharding = "{devices=[8,4]<=[32]}"} : (tensor<1x2xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[SHARD_TO_FULL]]

  %0 = sdy.reduce_scatter [{}, {"y"}] %arg0 out_sharding=<@mesh_2, [{"x"}, {"y"}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// Verify that the reduce_scatter is outlined into a private function representing the manual computation body,
// and that the operation inside uses manual sharding (preventing SPMD from inserting additional reductions).
// CHECK-LABEL:         func private @{{xla\.sdy\.inlinable_manual_computation_body(_[0-9]+)?}}
// CHECK-SAME{LITERAL}:     %arg0: tensor<1x8xf32> {mhlo.sharding = "{manual}"})
// CHECK-SAME{LITERAL}:     -> (tensor<1x2xf32> {mhlo.sharding = "{manual}"}) {
// CHECK-NEXT:            %[[REDUCE_SCATTER:.*]] = "stablehlo.reduce_scatter"(%arg0) <{
// CHECK-SAME:              channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
// CHECK-SAME{LITERAL}:     replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>,
// CHECK-SAME:              scatter_dimension = 1 : i64,
// CHECK-SAME:              use_global_device_ids}> ({
// CHECK-NEXT:            ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
// CHECK-NEXT:              %[[ADD:.*]] = stablehlo.add %arg1, %arg2 {mhlo.sharding = "{manual}"} : tensor<f32>
// CHECK-NEXT:              stablehlo.return %[[ADD]] : tensor<f32>
// CHECK-NEXT:            })
// CHECK-SAME:            {mhlo.sharding = "{manual}"}
// CHECK-SAME:            : (tensor<1x8xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:            return %[[REDUCE_SCATTER]] : tensor<1x2xf32>
// CHECK-NEXT:          }
