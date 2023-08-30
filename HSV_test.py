import tensorflow as tf
import numpy as np
from PIL import Image

img = np.array(Image.open("img0.png"))
img = img / 255.0
hsv = tf.slice(tf.keras.layers.Lambda(tf.image.rgb_to_hsv)(img), [0, 0, 1], [512, 512, 1]).numpy()
hsv = hsv*255
out = Image.fromarray(hsv)
out.show()