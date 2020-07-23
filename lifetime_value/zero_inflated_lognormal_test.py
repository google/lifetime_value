# Copyright 2019 The Lifetime Value Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Lint as: python3
# Dependency imports

from lifetime_value import zero_inflated_lognormal
import numpy as np
from scipy import stats
import tensorflow.compat.v1 as tf


# Absolute error tolerance in asserting array near.
_ERR_TOL = 1e-6

# softplus function that calculates log(1+exp(x))
_softplus = lambda x: np.log(1.0 + np.exp(x))

# sigmoid function that calculates 1/(1+exp(-x))
_sigmoid = lambda x: 1 / (1 + np.exp(-x))


class ZeroInflatedLognormalLossTest(tf.test.TestCase):

  def setUp(self):
    super(ZeroInflatedLognormalLossTest, self).setUp()
    self.logits = np.array([[.1, .2, .3], [.4, .5, .6]])
    self.labels = np.array([[0.], [1.5]])

  def zero_inflated_lognormal(self, labels, logits):
    positive_logits = logits[..., :1]
    loss_zero = _softplus(positive_logits)
    loc = logits[..., 1:2]
    scale = np.maximum(
        _softplus(logits[..., 2:]),
        np.sqrt(tf.keras.backend.epsilon()))
    log_prob_non_zero = stats.lognorm.logpdf(
        x=labels, s=scale, loc=0, scale=np.exp(loc))
    loss_non_zero = _softplus(-positive_logits) - log_prob_non_zero
    return np.mean(np.where(labels == 0., loss_zero, loss_non_zero), axis=-1)

  def test_loss_value(self):
    expected_loss = self.zero_inflated_lognormal(self.labels, self.logits)
    loss = zero_inflated_lognormal.zero_inflated_lognormal_loss(
        self.labels, self.logits)
    self.assertArrayNear(self.evaluate(loss), expected_loss, _ERR_TOL)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
