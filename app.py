from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "saved_model" / "1"
IMAGE_SIZE = (224, 224)


st.set_page_config(
    page_title="Animal Vision | Image Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #17221d; --muted: #66736b; --leaf: #2f6b4f; --mint: #e9f3ea; --gold: #e4a63d; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .hero { padding: 1.5rem 0 1rem; border-bottom: 1px solid #dbe5dc; margin-bottom: 1.5rem; }
    .eyebrow { color: var(--leaf); font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { font-size: clamp(2.2rem, 5vw, 4.4rem); line-height: .98; margin: .35rem 0 .8rem; }
    .hero p { color: var(--muted); font-size: 1.05rem; max-width: 650px; margin: 0; }
    .result-box { background: var(--mint); border-left: 5px solid var(--leaf); padding: 1rem 1.2rem; margin: 1rem 0 1.5rem; color: var(--ink); }
    .result-label { color: var(--muted) !important; font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; }
    .result-name { color: var(--ink) !important; font: 700 2rem 'Space Grotesk', sans-serif; margin-top: .2rem; }
    .hint { color: var(--muted); font-size: .9rem; }
    [data-testid="stMetric"] { background: #f5f8f4; border: 1px solid #dbe5dc; padding: .8rem 1rem; color: var(--ink); }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--ink) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    loaded = tf.saved_model.load(str(MODEL_PATH))
    return loaded.signatures["serving_default"]


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return tf.keras.applications.mobilenet.preprocess_input(array[None, ...])


def predict(image: Image.Image, inference):
    scores = inference(inputs=tf.constant(prepare_image(image)))
    probabilities = next(iter(scores.values())).numpy()[0]
    top_indices = np.argsort(probabilities)[::-1][:5]
    decoded = tf.keras.applications.mobilenet.decode_predictions(
        probabilities[None, ...], top=5
    )[0]
    return [(name.replace("_", " ").title(), float(score)) for _, name, score in decoded], probabilities


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Computer vision lab / 01</div>
      <h1>Animal Vision</h1>
      <p>Unggah sebuah gambar untuk melihat prediksi visual dari model dan tingkat keyakinannya.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Control panel")
    uploaded_file = st.file_uploader(
        "Pilih gambar", type=["jpg", "jpeg", "png", "webp"], label_visibility="visible"
    )
    st.divider()
    st.markdown("**Model aktif**")
    st.caption("MobileNet ImageNet")
    st.markdown("**Input**")
    st.caption("224 x 224 px · RGB · MobileNet preprocessing")
    st.info(
        "Artefak model saat ini menghasilkan 1.000 label ImageNet. Untuk classifier "
        "Cats / Dogs / Tigers, ekspor ulang model hasil training dari notebook."
    )

try:
    inference = load_model()
except Exception as error:
    st.error(f"Model tidak dapat dimuat: {error}")
    st.stop()

if uploaded_file is None:
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Mulai dari sebuah gambar")
        st.write("Gunakan panel di kiri untuk mengunggah foto JPG, PNG, atau WEBP.")
        st.markdown('<p class="hint">Hasil akan muncul di halaman ini tanpa mengirim gambar ke layanan eksternal.</p>', unsafe_allow_html=True)
    with right:
        st.metric("Model output", "1,000 labels")
        st.metric("Image size", "224 × 224")
else:
    image = Image.open(uploaded_file)
    predictions, _ = predict(image, inference)
    primary_label, primary_score = predictions[0]

    preview_col, result_col = st.columns([1.1, 1], gap="large")
    with preview_col:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)
        st.caption(f"{uploaded_file.name} · {image.width} × {image.height} px")
    with result_col:
        st.subheader("Prediction")
        st.markdown(
            f'<div class="result-box"><div class="result-label">Top match</div><div class="result-name">{primary_label}</div></div>',
            unsafe_allow_html=True,
        )
        confidence_col, labels_col = st.columns(2)
        confidence_col.metric("Confidence", f"{primary_score:.1%}")
        labels_col.metric("Candidates", "5")
        st.subheader("Top 5 probabilities")
        for label, score in predictions:
            st.markdown(f"**{label}** &nbsp; `{score:.1%}`")
            st.progress(score)

    st.divider()
    st.caption(
        "Catatan: dashboard ini memakai SavedModel yang tersedia di repository. "
        "Label di atas berasal dari ImageNet, bukan tiga kelas dataset proyek."
    )