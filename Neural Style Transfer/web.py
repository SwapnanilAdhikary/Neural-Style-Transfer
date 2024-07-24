import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


def load_img(img):
    img_bytes = img.read()

    img_array = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img_bgr, (256, 256))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = tf.expand_dims(img_normalized, axis=0)
    return img_tensor


def stylize_image(content_image, style_image):
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    stylized_image = tf.clip_by_value(stylized_image, 0.0, 1.0)
    return stylized_image.numpy()


def main():
    st.title("Styler")

    st.subheader("Upload Content Image")
    content_image = st.file_uploader("Choose a content image...", type=["jpg", "jpeg", "png"])
    st.subheader("Upload Style Image")
    style_image = st.file_uploader("Choose a style image...", type=["jpg", "jpeg", "png"])

    if content_image is not None and style_image is not None:

        content_img = load_img(content_image)
        style_img = load_img(style_image)

        st.subheader("Original Content Image")
        st.image(content_img[0].numpy(), use_column_width=True,width=300)

        st.subheader("Original Style Image")
        st.image(style_img[0].numpy(), use_column_width=True,width=300)

        if st.button("Stylize"):
            stylized_img = stylize_image(content_img, style_img)

            stylized_img = np.squeeze(stylized_img) * 255.0
            stylized_img = stylized_img.astype(np.uint8)
            stylized_img = cv2.cvtColor(stylized_img, cv2.COLOR_BGR2RGB)
            st.subheader("Stylized Image")
            st.image(stylized_img, use_column_width=True,width=300)


if __name__ == "__main__":
    main()
