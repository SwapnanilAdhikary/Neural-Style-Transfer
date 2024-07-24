import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


def load_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


def generate():
    content_image = load_img('xp.jpg')
    style_image = load_img('reference.jpg')
    stylized_images = model(tf.constant(content_image), tf.constant(style_image))[0]
    cv2.imwrite('generated_img2.jpg', cv2.cvtColor(np.squeeze(stylized_images) * 255, cv2.COLOR_BGR2RGB))

# plt.imshow(np.squeeze(stylized_images))
# plt.show()
