# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Reproduction test for b/510008212."""

import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test


class SelfAdjointEigReproTest(test.TestCase):

  def testDataRaceWithNaNs(self):
    # Shape large enough to trigger multithreading in LinearAlgebraOp
    shape = (2000, 5, 5)
    a = np.full(shape, np.nan, dtype=np.float32)
    matrix_tensor = constant_op.constant(a)

    with self.session():
      e, v = linalg_ops.self_adjoint_eig(matrix_tensor)
      e_val, v_val = self.evaluate([e, v])
      self.assertTrue(np.isnan(e_val).all())
      self.assertTrue(np.isnan(v_val).all())


if __name__ == "__main__":
  test.main()
