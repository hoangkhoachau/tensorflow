module attributes {tf.versions = {producer = 888 : i32}} {
  func.func @main(%arg0: tensor<1x3xi32>, %arg1: tensor<3x1xi32>) -> (tensor<1x1xi32>, tensor<1x1xi32>) {
    %0 = "tf_device.cluster_func"(%arg0, %arg1) {
      _replication_info = "cluster",
      func = @cluster_func,
      num_cores_per_replica = 1,
      step_marker_location = "",
      input_sharding_configuration = ["", ""],
      output_sharding_configuration = [""],
      use_spmd_for_xla_partitioning = false
    } : (tensor<1x3xi32>, tensor<3x1xi32>) -> tensor<1x1xi32>
    %1 = "tf_device.cluster_func"(%arg0, %arg1) {
      _replication_info = "cluster",
      func = @cluster_func,
      num_cores_per_replica = 1,
      step_marker_location = "",
      input_sharding_configuration = ["", ""],
      output_sharding_configuration = [""],
      use_spmd_for_xla_partitioning = false
    } : (tensor<1x3xi32>, tensor<3x1xi32>) -> tensor<1x1xi32>
    return %0, %1 : tensor<1x1xi32>, tensor<1x1xi32>
  }

  func.func @cluster_func(%arg0: tensor<1x3xi32>, %arg1: tensor<3x1xi32>) -> tensor<1x1xi32> {
    %0 = "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : (tensor<1x3xi32>, tensor<3x1xi32>) -> tensor<1x1xi32>
    return %0 : tensor<1x1xi32>
  }
}
