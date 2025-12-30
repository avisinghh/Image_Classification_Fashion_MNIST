#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fashion-MNIST Classifier — Streamlit UI (enhanced)
- Auto-reloads updated model (cache keyed by file mtime)
- Sidebar shows model name & param count
- Safe large-image guard on upload
- Sanity 3×3 sample grid
- Test-set evaluation tab: accuracy, confusion matrix, classification report
Run: conda activate ML_DL && streamlit run fashion_predict_app.py
"""

import os, io, numpy as np, pandas as pd
from datetime import datetime
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# Page & global styling
# =========================
st.set_page_config(page_title="Fashion-MNIST Classifier", page_icon="👗", layout="wide")
st.markdown(
    """
    <style>
      .big-pred {font-size: 1.4rem; font-weight: 700; margin: 0.25rem 0;}
      .subtle {opacity: 0.75;}
      .badge {display:inline-block; padding: .25rem .6rem; border-radius: 999px; background: #EEF2FF; color:#3730A3; font-weight:600;}
      .good  {color:#1a7f37;}
      .bad   {color:#b3261e;}
      .pill  {padding:.25rem .6rem;border-radius:999px; background:#F1F5F9;}
      .card  {border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:white;}
      .st-emotion-cache-1b2oi0j img {border-radius:12px;}
      .mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("👗 Fashion-MNIST Classifier")
st.caption("Upload any image — we convert it to **28×28 grayscale** (Fashion-MNIST format) and predict its class. Use the sidebar to tweak preprocessing.")

# =========================
# Device info
# =========================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
st.info(f"TensorFlow {tf.__version__} • GPU(s) detected: {len(gpus)}")

# =========================
# Sidebar controls
# =========================
st.sidebar.header("⚙️ Settings")
MODEL_PATH = st.sidebar.text_input("Model path", value="fashion_cnn.keras")

# Model file info + reload controls
def _file_info(path: str):
    if not os.path.exists(path):
        return "—", 0.0, 0
    mtime = os.path.getmtime(path)
    size = os.path.getsize(path)
    ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    return ts, mtime, size

ts, mtime, fsize = _file_info(MODEL_PATH)
st.sidebar.caption(
    f"Last modified: **{ts}**  •  Size: **{fsize/1_048_576:.2f} MB**"
    if ts != "—" else "Model file not found."
)

col_rb1, col_rb2 = st.sidebar.columns(2)
with col_rb1:
    if st.button("🔄 Reload model"):
        st.cache_resource.clear()
        st.rerun()
with col_rb2:
    auto_reload = st.toggle("Auto-reload", value=True, help="Reload if file timestamp changes on rerun.")

TOP_K = st.sidebar.slider("Show top-K classes", 3, 10, 5, 1)
st.sidebar.divider()
st.sidebar.subheader("Preprocessing")
pp_letterbox = st.sidebar.selectbox("Resize mode", ["Fit with padding (letterbox)", "Center crop then resize"], index=0)
pp_invert    = st.sidebar.toggle("Invert colors (photo → MNIST style)", value=False, help="Fashion-MNIST is light foreground on dark bg; invert helps real photos.")
pp_equalize  = st.sidebar.toggle("Auto-contrast / equalize", value=True, help="Improves contrast for dim images")
pp_normalize = st.sidebar.toggle("Normalize to [0,1]", value=True)

st.sidebar.divider()
st.sidebar.subheader("Try samples")
sample_mode = st.sidebar.radio("Pick sample source", ["None", "Random", "Pick by class"], index=0, horizontal=False)
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
sample_class = st.sidebar.selectbox("Class", CLASS_NAMES, index=7, disabled=(sample_mode!="Pick by class"))

# =========================
# Model loader (cached with file mtime)
# =========================
@st.cache_resource(show_spinner=True)
def load_model(path: str, file_mtime: float):
    """
    Cache key includes the file mtime, so updating the model file
    automatically invalidates the cache on the next rerun.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return tf.keras.models.load_model(path)

# Optional auto-reload: clear cache if mtime changed since last load
if auto_reload:
    prev_mtime = st.session_state.get("_loaded_model_mtime", None)
    if prev_mtime is not None and mtime != prev_mtime:
        st.cache_resource.clear()
st.session_state["_loaded_model_mtime"] = mtime

try:
    model = load_model(MODEL_PATH, mtime)
    total_params = model.count_params()
    st.success(f"Loaded model: `{os.path.basename(MODEL_PATH)}` (mtime key: {mtime})")
    st.sidebar.success(f"Loaded: **{os.path.basename(MODEL_PATH)}** • Params: **{total_params:,}**")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# =========================
# Data utilities
# =========================
TARGET_SIZE = (28, 28)

def letterbox_to_square(img: Image.Image, fill=0) -> Image.Image:
    """Pad to square, keeping aspect ratio."""
    w, h = img.size
    side = max(w, h)
    bg = Image.new("L", (side, side), color=fill)
    offset = ((side - w) // 2, (side - h) // 2)
    bg.paste(img, offset)
    return bg

@st.cache_data(show_spinner=False)
def get_random_test_sample(class_name: str|None=None):
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    if class_name is None:
        idx = np.random.randint(0, len(x_te))
    else:
        cid = CLASS_NAMES.index(class_name)
        idxs = np.where(y_te == cid)[0]
        idx = np.random.choice(idxs)
    img = Image.fromarray(x_te[idx])
    true = CLASS_NAMES[int(y_te[idx])]
    return img, true

@st.cache_data(show_spinner=False)
def load_testset():
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    x_te = (x_te.astype("float32")/255.0)[..., None]
    return x_te, y_te

def preprocess_pil(img: Image.Image,
                   invert=False, equalize=True, normalize=True,
                   mode="Fit with padding (letterbox)") -> np.ndarray:
    """Returns (1, 28, 28, 1) float32 ready for Keras."""
    img = img.convert("L")
    if mode.startswith("Fit"):
        img28 = letterbox_to_square(img)
        img28 = img28.resize(TARGET_SIZE, Image.BILINEAR)
    else:
        # center crop to square then resize
        w, h = img.size
        side = min(w, h)
        left = (w - side)//2; top = (h - side)//2
        img28 = img.crop((left, top, left+side, top+side)).resize(TARGET_SIZE, Image.BILINEAR)

    if invert:   img28 = ImageOps.invert(img28)
    if equalize: img28 = ImageOps.autocontrast(img28)

    arr = np.array(img28).astype("float32")
    if normalize: arr = arr / 255.0
    arr = arr[..., None]           # (28,28,1)
    arr = arr[None, ...]           # (1,28,28,1)
    return arr

def predict(arr: np.ndarray):
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return idx, probs

# =========================
# Tabs
# =========================
tab_pred, tab_eval, tab_info = st.tabs(["🔮 Predict", "📊 Evaluate on Test Set", "ℹ️ Model Info"])

# ------------------------- Predict Tab -------------------------
with tab_pred:
    c1, c2 = st.columns([7, 5], gap="large")

    with c1:
        st.subheader("1) Choose an image")
        up = st.file_uploader("Drop an image (JPG/PNG/BMP/WEBP). Any size/color — we’ll preprocess to 28×28.",
                              type=["jpg","jpeg","png","bmp","webp"])
        img_raw, true_hint = None, None

        if sample_mode == "Random":
            img_raw, true_hint = get_random_test_sample(None)
            st.caption("Using a random Fashion-MNIST test image.")
        elif sample_mode == "Pick by class":
            img_raw, true_hint = get_random_test_sample(sample_class)
            st.caption(f"Using a random **{sample_class}** from Fashion-MNIST test set.")
        elif up is not None:
            # Safe large-image guard
            img_raw = Image.open(io.BytesIO(up.read()))
            max_side = 4096
            if max(img_raw.size) > max_side:
                scale = max_side / float(max(img_raw.size))
                new_size = (int(img_raw.size[0]*scale), int(img_raw.size[1]*scale))
                img_raw = img_raw.resize(new_size, Image.BILINEAR)
            st.caption("Uploaded image")

        if img_raw is not None:
            st.image(img_raw, caption="Original", use_container_width=True)

    with c2:
        st.subheader("2) Preview preprocessing")
        if img_raw is None:
            st.info("Upload or select a sample to preview preprocessing.")
        else:
            arr = preprocess_pil(
                img_raw,
                invert=pp_invert,
                equalize=pp_equalize,
                normalize=pp_normalize,
                mode=pp_letterbox,
            )
            preview = (arr[0, ..., 0] * (255 if pp_normalize else 1)).astype("uint8")
            st.image(preview, caption="Preprocessed • 28×28 (what the model sees)", clamp=True, width=240)
            st.caption(
                f"<span class='pill'>invert={pp_invert}</span> "
                f"<span class='pill'>equalize={pp_equalize}</span> "
                f"<span class='pill'>normalize={pp_normalize}</span> "
                f"<span class='pill'>{pp_letterbox}</span>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("3) Predict")

    if img_raw is None:
        st.warning("Select a sample or upload an image above.")
    else:
        pred_idx, probs = predict(arr)
        pred_label = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx])

        # Header result
        st.markdown(f"<div class='big-pred'>Prediction: <span class='badge'>{pred_label}</span></div>", unsafe_allow_html=True)
        st.write(f"Confidence: **{conf:.3f}**")
        if true_hint:
            color = "good" if true_hint == pred_label else "bad"
            st.markdown(f"<span class='{color}'>True label (sample): {true_hint}</span>", unsafe_allow_html=True)

        # Top-K table + bars
        df = pd.DataFrame({"class": CLASS_NAMES, "probability": probs})
        df_sorted = df.sort_values("probability", ascending=False).reset_index(drop=True)

        t1, t2 = st.columns([5, 7], gap="large")
        with t1:
            st.dataframe(df_sorted.head(TOP_K), use_container_width=True)

        with t2:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            sns.barplot(data=df_sorted.head(TOP_K), x="probability", y="class", ax=ax)
            ax.set_xlim(0, 1); ax.set_xlabel("Probability"); ax.set_ylabel("")
            ax.set_title("Top-K probabilities")
            st.pyplot(fig, clear_figure=True)

        # Optional download of preprocessed tile
        st.download_button(
            "⬇️ Download preprocessed 28×28 image",
            data=Image.fromarray(preview).tobytes(),
            file_name="preprocessed_28x28.raw",
            mime="application/octet-stream",
            help="Raw bytes (uint8, 28×28)."
        )

    # Sanity 3×3 sample grid
    with st.expander("🔍 See a 3×3 grid of random test samples"):
        x_te, y_te = load_testset()
        raw = (x_te[..., 0] * 255).astype("uint8")
        idxs = np.random.choice(len(x_te), 9, replace=False)
        cols = st.columns(3)
        for r in range(3):
            for c in range(3):
                i = idxs[r*3+c]
                cols[c].image(raw[i], caption=f"True: {CLASS_NAMES[int(y_te[i])]}", clamp=True, use_container_width=True)

# ------------------------- Evaluate Tab -------------------------
with tab_eval:
    st.subheader("Evaluate on Fashion-MNIST Test Set")
    x_te, y_te = load_testset()
    with st.spinner("Scoring test set..."):
        loss, acc = model.evaluate(x_te, y_te, verbose=0)
        y_prob = model.predict(x_te, verbose=0)
        y_pred = y_prob.argmax(1)

    st.success(f"🎯 Test Accuracy: **{acc:.4f}**  •  Loss: **{loss:.4f}**")

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
    ax_cm.set_title("Confusion Matrix (FMNIST Test)")
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    ax_cm.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax_cm.set_yticklabels(CLASS_NAMES, rotation=0, fontsize=8)
    st.pyplot(fig_cm, clear_figure=True)

    # Classification report
    report = classification_report(y_te, y_pred, target_names=CLASS_NAMES, digits=4)
    st.markdown("**Classification Report**")
    st.code(report, language="text")

# ------------------------- Info Tab -------------------------
with tab_info:
    st.subheader("Model summary")
    with st.expander("Show Keras summary"):
        # Capture summary text
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        st.code("\n".join(stringlist), language="text")

    st.subheader("About this app")
    st.markdown(
        """
        - Expects **28×28 grayscale** images (preprocessed automatically).
        - Trained on **Fashion-MNIST** (10 classes).
        - Preprocessing controls help adapt real photos:
            - **Invert**: many real photos are dark-on-light; FMNIST is light-on-dark.
            - **Equalize**: improves contrast for dim images.
            - **Letterbox** vs **Center crop**: choose how we map to square before resizing.
        """
    )
