load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_oss")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_fused_conv_rewriter_test_common",
    testonly = True,
    srcs = ["cudnn_fused_conv_rewriter_test.cc"],
    compatible_with = [],
    tags = ["gpu"],
    deps = [
        "//xla:comparison_util",
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/transforms:conv_rewriter",
        "//xla/backends/gpu/transforms:cudnn_fused_conv_rewriter",
        "//xla/deprecated/backends/gpu/tests:gpu_codegen_test",
        "//xla/deprecated/tests:hlo_test_base_legacy",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/hlo/transforms/simplifiers:convert_mover",
        "//xla/hlo/transforms/simplifiers:hlo_constant_folding",
        "//xla/hlo/transforms/simplifiers:reshape_mover",
        "//xla/service:hlo_module_config",
        "//xla/service:pattern_matcher",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:semantic_version",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/protobuf:dnn_proto_cc",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "cudnn_fused_conv_rewriter_test",
    srcs = [],
    backend_tags = {
        "gpu_a100": [
            "noasan",
            "nomsan",
        ],
    },
    backends = [
        "a100",
        "amdgpu_any",
    ] + if_oss(["nvgpu_any"]),
    shard_count = 10,
    use_legacy_runtime = True,
    deps = [
        ":cudnn_fused_conv_rewriter_test_common",
    ],
)

xla_test(
    name = "cudnn_fused_conv_rewriter_autotune_disabled_test",
    srcs = [],
    args = ["--xla_gpu_autotune_level=0"],
    backends = [
        "a100",
        "amdgpu_any",
    ],
    shard_count = 4,
    use_legacy_runtime = True,
    deps = [
        ":cudnn_fused_conv_rewriter_test_common",
    ],
)

xla_test(
    name = "cudnn_norm_rewriter_test",
    srcs = ["cudnn_norm_rewriter_test.cc"],
    backends = ["gpu"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    use_legacy_runtime = True,
    deps = [
        "//xla:error_spec",
        "//xla/backends/gpu/transforms:cudnn_norm_rewriter",
        "//xla/deprecated/backends/gpu/tests:gpu_codegen_test",
        "//xla/hlo/testlib:filecheck",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudnn_header",
    ]),
)

xla_test(
    name = "dot_dimension_sorter_test",
    srcs = ["dot_dimension_sorter_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        "//xla:error_spec",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/transforms:dot_dimension_sorter",
        "//xla/deprecated/backends/gpu/tests:gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "gemm_broadcast_folding_rewriter_test",
    srcs = ["gemm_broadcast_folding_rewriter_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        "//xla:error_spec",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/transforms:gemm_broadcast_folding_rewriter",
        "//xla/backends/gpu/transforms:gemm_rewriter",
        "//xla/deprecated/backends/gpu/tests:gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:semantic_version",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "gemm_rewriter_allocation_test",
    srcs = ["gemm_rewriter_allocation_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        "//xla/deprecated/backends/gpu/tests:hlo_legacy_gpu_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:stream_executor_address_allocator",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "scaled_dot_rewriter_test",
    srcs = ["scaled_dot_rewriter_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        "//xla:error_spec",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/transforms:scaled_dot_rewriter",
        "//xla/deprecated/tests:hlo_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
