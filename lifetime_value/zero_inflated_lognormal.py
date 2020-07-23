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
"""Zero-inflated lognormal loss for lifetime value prediction."""
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def zero_inflated_lognormal_pred(logits: tf.Tensor) -> tf.Tensor:
  """Calculates predicted mean of zero inflated lognormal logits.

  Arguments:
    logits: [batch_size, 3] tensor of logits.

  Returns:
    preds: [batch_size, 1] tensor of predicted mean.
  """
  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  positive_probs = tf.keras.backend.sigmoid(logits[..., :1])
  loc = logits[..., 1:2]
  scale = tf.keras.backend.softplus(logits[..., 2:])
  preds = (
      positive_probs *
      tf.keras.backend.exp(loc + 0.5 * tf.keras.backend.square(scale)))
  return preds


def zero_inflated_lognormal_loss(labels: tf.Tensor,
                                 logits: tf.Tensor) -> tf.Tensor:
  """Computes the zero inflated lognormal loss.

  Usage with tf.keras API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=zero_inflated_lognormal)
  ```

  Arguments:
    labels: True targets, tensor of shape [batch_size, 1].
    logits: Logits of output layer, tensor of shape [batch_size, 3].

  Returns:
    Zero inflated lognormal loss value.
  """
  labels = tf.convert_to_tensor(labels, dtype=tf.float32)
  positive = tf.cast(labels > 0, tf.float32)

  logits = tf.convert_to_tensor(logits, dtype=tf.float32)
  logits.shape.assert_is_compatible_with(
      tf.TensorShape(labels.shape[:-1].as_list() + [3]))

  positive_logits = logits[..., :1]
  classification_loss = tf.keras.losses.binary_crossentropy(
      y_true=positive, y_pred=positive_logits, from_logits=True)

  loc = logits[..., 1:2]
  scale = tf.math.maximum(
      tf.keras.backend.softplus(logits[..., 2:]),
      tf.math.sqrt(tf.keras.backend.epsilon()))
  safe_labels = positive * labels + (
      1 - positive) * tf.keras.backend.ones_like(labels)
  regression_loss = -tf.keras.backend.mean(
      positive * tfd.LogNormal(loc=loc, scale=scale).log_prob(safe_labels),
      axis=-1)

  return classification_loss + regression_loss
