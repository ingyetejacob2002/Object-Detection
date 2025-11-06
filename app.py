import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os

st.title("🎯 YOLOv8 Object Detection App")
st.write("Upload an image and let YOLOv8 detect objects in it!")

model = YOLO("yolov8n.pt")  # Pretrained YOLOv8 model

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.write("🔍 Detecting objects...")
    results = model(tmp_path)

    result_image = results[0].plot()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    st.image(result_image, caption="Detection Result", use_column_width=True)

    st.subheader("🧾 Detected Objects:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        st.write(f"- **{cls_name}** ({conf*100:.1f}% confidence)")

    os.remove(tmp_path)
