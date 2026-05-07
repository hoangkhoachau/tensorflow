import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables


class KernelCacheGrowthTest(test.TestCase):

  def _network(self, x):
    conv1 = nn_ops.conv2d(
        x, self._kernel1, strides=[1, 1, 1, 1], padding="SAME")
    conv2 = nn_ops.conv2d(
        nn_ops.relu(conv1), self._kernel2, strides=[1, 1, 1, 1],
        padding="SAME")
    pooled = math_ops.reduce_mean(nn_ops.relu(conv2), axis=[1, 2])
    return math_ops.matmul(pooled, self._projection)

  def _run_and_collect_stats(self, label, fn, iterations=20):
    ctx = context.context()
    ctx.clear_kernel_cache()
    stats = [ctx.get_cache_stats()]
    print("%s_CACHE_STATS_0 %s" % (label, stats[-1]))
    for i in range(iterations):
      fn()
      stats.append(ctx.get_cache_stats())
      print("%s_CACHE_STATS_%d %s" % (label, i + 1, stats[-1]))
    return stats

  def _kernel_cache_delta(self, stats):
    return stats[-1]["kernel_cache_size"] - stats[0]["kernel_cache_size"]

  @test_util.run_in_graph_and_eager_modes
  def test_kernel_cache_growth_paths(self):
    if not context.executing_eagerly():
      return
    ctx = context.context()
    self._kernel1 = variables.Variable(
        array_ops.ones((3, 3, 3, 16), dtype=dtypes.float32))
    self._kernel2 = variables.Variable(
        array_ops.ones((3, 3, 16, 16), dtype=dtypes.float32))
    self._projection = variables.Variable(
        array_ops.ones((16, 8), dtype=dtypes.float32))
    np_x = np.zeros((1, 32, 32, 3), dtype=np.float32)
    tensor_x = array_ops.zeros((1, 32, 32, 3), dtype=dtypes.float32)

    direct_numpy_stats = self._run_and_collect_stats(
        "DIRECT_NUMPY",
        lambda: self._network(np_x).numpy())

    after_direct = ctx.get_cache_stats()
    ctx.clear_kernel_cache()
    after_clear = ctx.get_cache_stats()
    print("AFTER_DIRECT_CLEAR_BEFORE %s" % after_direct)
    print("AFTER_DIRECT_CLEAR_AFTER %s" % after_clear)

    direct_tensor_stats = self._run_and_collect_stats(
        "DIRECT_TENSOR",
        lambda: self._network(tensor_x).numpy())

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec((1, 32, 32, 3), dtype=dtypes.float32)
    ])
    def compiled_call(x):
      return self._network(x)

    tf_function_stats = self._run_and_collect_stats(
        "TF_FUNCTION",
        lambda: compiled_call(tensor_x).numpy())

    print("CACHE_DELTA_DIRECT_NUMPY %d" %
          self._kernel_cache_delta(direct_numpy_stats))
    print("CACHE_DELTA_DIRECT_TENSOR %d" %
          self._kernel_cache_delta(direct_tensor_stats))
    print("CACHE_DELTA_TF_FUNCTION %d" %
          self._kernel_cache_delta(tf_function_stats))

    self.assertGreater(self._kernel_cache_delta(direct_numpy_stats), 0)
    self.assertLess(after_clear["kernel_cache_size"],
                    after_direct["kernel_cache_size"])


if __name__ == "__main__":
  test.main()
