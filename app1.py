import streamlit as st

import numpy as np

st.title("Pipeline Structural Health Monitoring System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 10, 60)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Overlay cracks
    image_rgb[edges > 0] = [255,0,0]

    st.image(image_rgb, caption="Detected Cracks")

    # Result download
    cv2.imwrite("result.jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    with open("result.jpg", "rb") as f:
        st.download_button("Download Result", f, "result.jpg")

    # Result message
    if np.sum(edges) > 0:
        st.success("Crack Detected")
    else:
        st.info("No crack detected")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image_rgb, caption="Processed Image")

    with col2:
        st.image(edges, caption="Detected Edges")
