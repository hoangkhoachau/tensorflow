load("//xla/tests:build_defs.bzl", "xla_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
    transitive_visibility = "//xla:restricted",
)

xla_test(
    name = "fusion_emitter_parametrized_test",
    srcs = ["fusion_emitter_parametrized_test.cc"],
    backends = [
        "a100",
        "h100",
        "b200",
        "amdgpu_any",
    ],
    tags = ["no_mac"],
    use_legacy_runtime = True,
    deps = [
        "//xla:comparison_util",
        "//xla:error_spec",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/codegen/triton:support",
        "//xla/backends/gpu/codegen/triton:test_utils",
        "//xla/backends/gpu/tests/restricted:gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)
