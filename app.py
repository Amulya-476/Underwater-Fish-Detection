from pathlib import Path
import io

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -------------------- Configuration --------------------
st.set_page_config(
    page_title="Underwater Fish Detection",
    page_icon="🐟",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "weights" / "best.pt"

st.title("🐟 Underwater Fish Detection")
st.write(
    "Upload an underwater image to detect and identify fish using the "
    "trained YOLOv8 model."
)

# -------------------- Load trained model --------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Make sure weights/best.pt is inside the project folder."
        )
    return YOLO(str(MODEL_PATH))

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# -------------------- User controls --------------------
with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Lower this value if some visible fish are not detected."
    )
    image_size = st.selectbox("Image size", [640, 800, 1024], index=0)
    st.caption("A lower confidence threshold may detect more fish, but can also add false detections.")

uploaded_file = st.file_uploader(
    "Upload an underwater fish image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    with st.spinner("Detecting fish..."):
        result = model.predict(
            source=image,
            conf=confidence,
            imgsz=image_size,
            iou=0.5,
            max_det=300,
            verbose=False
        )[0]

    # Ultralytics returns an annotated BGR image; convert it for Streamlit.
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated_image = Image.fromarray(annotated_rgb)

    boxes = result.boxes
    total_detections = 0 if boxes is None else len(boxes)

    with right:
        st.subheader("Detection result")
        st.image(annotated_image, use_container_width=True)

    st.divider()
    st.subheader("Monitoring summary")
    st.metric("Total fish detections", total_detections)

    if total_detections:
        names = model.names
        rows = []
        for box in boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            rows.append({
                "Species": names[class_id],
                "Confidence": f"{score * 100:.1f}%",
                "Bounding box (x1, y1, x2, y2)": str(xyxy)
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)

        output = io.BytesIO()
        annotated_image.save(output, format="PNG")
        st.download_button(
            "Download detected image",
            data=output.getvalue(),
            file_name="fish_detection_result.png",
            mime="image/png"
        )
    else:
        st.info(
            "No fish detected at this confidence threshold. "
            "Try lowering the confidence setting or uploading a clearer image."
        )
else:
    st.info("Upload an image to begin fish detection.")
