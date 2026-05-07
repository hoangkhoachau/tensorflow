import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import test_util
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops


class KernelCacheGrowthTest(test.TestCase):

  def _model(self):
    inputs = layers.Input(shape=(32, 32, 3))
    x = inputs
    for _ in range(4):
      x = layers.Conv2D(16, 3, padding="same", use_bias=False)(x)
      x = layers.BatchNormalization(scale=False)(x)
      x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(8, use_bias=False)(x)
    return models.Model(inputs, outputs)

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
    model = self._model()
    np_x = np.zeros((1, 32, 32, 3), dtype=np.float32)
    tensor_x = array_ops.zeros((1, 32, 32, 3))

    direct_numpy_stats = self._run_and_collect_stats(
        "DIRECT_NUMPY",
        lambda: model(np_x, training=False).numpy())

    after_direct = ctx.get_cache_stats()
    ctx.clear_kernel_cache()
    after_clear = ctx.get_cache_stats()
    print("AFTER_DIRECT_CLEAR_BEFORE %s" % after_direct)
    print("AFTER_DIRECT_CLEAR_AFTER %s" % after_clear)

    direct_tensor_stats = self._run_and_collect_stats(
        "DIRECT_TENSOR",
        lambda: model(tensor_x, training=False).numpy())

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec((1, 32, 32, 3), dtype=tensor_x.dtype)
    ])
    def compiled_call(x):
      return model(x, training=False)

    tf_function_stats = self._run_and_collect_stats(
        "TF_FUNCTION",
        lambda: compiled_call(tensor_x).numpy())

    predict_on_batch_stats = self._run_and_collect_stats(
        "PREDICT_ON_BATCH",
        lambda: model.predict_on_batch(np_x))

    print("CACHE_DELTA_DIRECT_NUMPY %d" %
          self._kernel_cache_delta(direct_numpy_stats))
    print("CACHE_DELTA_DIRECT_TENSOR %d" %
          self._kernel_cache_delta(direct_tensor_stats))
    print("CACHE_DELTA_TF_FUNCTION %d" %
          self._kernel_cache_delta(tf_function_stats))
    print("CACHE_DELTA_PREDICT_ON_BATCH %d" %
          self._kernel_cache_delta(predict_on_batch_stats))

    self.assertGreater(self._kernel_cache_delta(direct_numpy_stats), 0)
    self.assertLess(after_clear["kernel_cache_size"],
                    after_direct["kernel_cache_size"])


if __name__ == "__main__":
  test.main()
