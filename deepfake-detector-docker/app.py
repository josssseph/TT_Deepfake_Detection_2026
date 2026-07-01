"""Interfaz Streamlit para analizar un video mediante el modelo seleccionado."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from deepfake_app.errors import DeepfakeAppError
from deepfake_app.inference import analyze_video
from deepfake_app.resources import load_resources
from deepfake_app.types import InferenceResult


st.set_page_config(
    page_title="Detector de Deepfakes",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 920px; padding-top: 2rem; padding-bottom: 3rem;}
        h1, h2, h3, p, label {letter-spacing: 0;}
        .app-kicker {font-size: 0.82rem; font-weight: 700; color: #475467; margin-bottom: 0.25rem;}
        .result-panel {border: 1px solid #d0d5dd; border-left-width: 6px; padding: 1rem 1.1rem; border-radius: 6px; margin: 0.6rem 0 1rem;}
        .result-real {border-left-color: #067647; background: #ecfdf3;}
        .result-fake {border-left-color: #b42318; background: #fef3f2;}
        .result-label {font-size: 1.35rem; font-weight: 750; color: #101828;}
        .result-caption {font-size: 0.9rem; color: #475467; margin-top: 0.2rem;}
        div[data-testid="stMetric"] {border: 1px solid #e4e7ec; padding: 0.75rem; border-radius: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_resources():
    return load_resources()


def render_result(result: InferenceResult) -> None:
    is_fake = result.label.lower() == "fake"
    panel_class = "result-fake" if is_fake else "result-real"
    st.markdown(
        f"""
        <div class="result-panel {panel_class}">
            <div class="result-label">Clasificacion: {result.label}</div>
            <div class="result-caption">Decision obtenida con el umbral operativo fijado para el modelo.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_column, threshold_column, time_column = st.columns(3)
    score_column.metric("Puntaje Fake", f"{result.fake_score * 100:.2f} %")
    threshold_column.metric("Umbral", f"{result.threshold:.3f}")
    time_column.metric("Tiempo", f"{result.elapsed_seconds:.1f} s")

    diagnostics = result.diagnostics
    if diagnostics.padded_positions > 0:
        st.warning(
            f"Se detectaron {diagnostics.valid_faces} rostros validos y se "
            f"rellenaron {diagnostics.padded_positions} posiciones repitiendo "
            "el ultimo rostro disponible."
        )

    with st.expander("Detalles tecnicos"):
        st.write(f"Dispositivo: `{result.device}`")
        st.write(f"Fotogramas del video: `{diagnostics.total_video_frames}`")
        st.write(f"Posiciones decodificadas: `{diagnostics.decoded_positions}`")
        st.write(f"Rostros validos: `{diagnostics.valid_faces}`")
        st.write(f"Posiciones usadas por el modelo: `{diagnostics.selected_positions}`")
        st.write(
            "Logits Real/Fake: "
            f"`({result.logits[0]:.5f}, {result.logits[1]:.5f})`"
        )

    st.info(
        "El Puntaje Fake es una salida del modelo y no una probabilidad "
        "calibrada. El sistema fue evaluado con FaceForensics++ C23 y no "
        "constituye una certificacion de autenticidad."
    )


st.markdown('<div class="app-kicker">PROTOTIPO DE INFERENCIA LOCAL</div>', unsafe_allow_html=True)
st.title("Detector de Deepfakes")
st.write("Selecciona un video MP4 para obtener una clasificacion a nivel de video.")

try:
    with st.spinner("Cargando modelos de inferencia..."):
        resources = get_resources()
except DeepfakeAppError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"No fue posible iniciar los modelos: {exc}")
    st.stop()

uploaded = st.file_uploader(
    "Video MP4",
    type=["mp4"],
    accept_multiple_files=False,
    max_upload_size=500,
    width="stretch",
)

if uploaded is not None:
    video_bytes = uploaded.getvalue()
    upload_digest = hashlib.sha256(video_bytes).hexdigest()
    if st.session_state.get("upload_digest") != upload_digest:
        st.session_state["upload_digest"] = upload_digest
        st.session_state.pop("inference_result", None)

    st.video(video_bytes)
    analyze = st.button("Analizar video", type="primary", width="stretch")

    if analyze:
        progress = st.progress(0, text="Preparando analisis")

        def update_progress(message: str, fraction: float) -> None:
            value = max(0, min(100, int(round(fraction * 100))))
            progress.progress(value, text=message)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temporary:
                temporary.write(video_bytes)
                temporary_path = Path(temporary.name)

            result = analyze_video(
                temporary_path,
                resources,
                progress_callback=update_progress,
            )
            st.session_state["inference_result"] = result
            progress.empty()
        except DeepfakeAppError as exc:
            progress.empty()
            st.error(str(exc))
        except Exception as exc:
            progress.empty()
            st.error(f"El analisis no pudo completarse: {exc}")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    stored_result = st.session_state.get("inference_result")
    if stored_result is not None:
        render_result(stored_result)
