"""Fail fast unless the configured TensorFlow runtime can train on a GPU."""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("GPU not detected; refusing to silently run full training on CPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("TensorFlow:", tf.__version__, "GPU:", gpus, flush=True)
with tf.device("/GPU:0"):
    kernel = tf.Variable(tf.random.normal([3, 3, 3, 8]))
    x = tf.random.normal([2, 64, 64, 3])
    with tf.GradientTape() as tape:
        output = tf.nn.conv2d(x, kernel, 1, "SAME")
        loss = tf.reduce_mean(output ** 2)
    grad = tape.gradient(loss, kernel)
    tf.keras.optimizers.Adam().apply_gradients([(grad, kernel)])
    tf.debugging.assert_all_finite(grad, "GPU gradient contains NaN/Inf")
    assert "GPU" in output.device, output.device
    print("GPU forward/backward/optimizer passed:", float(loss), output.device, flush=True)
