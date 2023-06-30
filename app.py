import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
model = tf.keras.models.load_model("model.h5")


def predict(image):
    image = image.resize((100, 81))  # Resize image to match the model's input size
    image_array = np.asarray(image) / 255.0  # Convert image to array and normalize

    # Make prediction using the loaded model
    points_list = model.predict(image_array.reshape(1, 81, 100, 3)).astype('int')[0]

    # Process and scale the predicted points to match the original image size
    x_ratio = 1.05 * (178 / 100)
    y_ratio = 1.085 * (100 / 81)

    points_list[0] = int(points_list[0] * x_ratio)
    points_list[2] = int(points_list[2] * x_ratio)
    points_list[4] = int(points_list[4] * x_ratio)
    points_list[6] = int(points_list[6] * x_ratio)
    points_list[8] = int(points_list[8] * x_ratio)

    points_list[1] = int(points_list[1] * y_ratio)
    points_list[3] = int(points_list[3] * y_ratio)
    points_list[5] = int(points_list[5] * y_ratio)
    points_list[7] = int(points_list[7] * y_ratio)
    points_list[9] = int(points_list[9] * y_ratio)

    return points_list

def main():
    st.title("Image Region Detection")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        points_list = predict(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Detected Region Points:")
        st.write(points_list)
        st.image(image_with_box(image, points_list), caption="Detected Region", use_column_width=True)


def image_with_box(image, points_list):
    width = abs(points_list[0] - points_list[8] - 60)
    height = abs(points_list[1] - points_list[9] - 75)
    rect = Image.new("RGBA", (width, height), (0, 255, 0, 0))
    image.paste(rect, (points_list[0] - 30, points_list[1] - 40), mask=rect)
    return image


if __name__ == '__main__':
    main()
