"""Interfaz Streamlit para analizar un video mediante el modelo seleccionado."""

from __future__ import annotations

import hashlib
import html
import tempfile
from pathlib import Path

import streamlit as st

from deepfake_app.errors import DeepfakeAppError
from deepfake_app.inference import analyze_video
from deepfake_app.resources import load_resources
from deepfake_app.types import InferenceResult


st.set_page_config(
    page_title="Detector de Deepfakes",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #18212b;
            --muted: #5d6874;
            --line: #d9dee5;
            --surface: #ffffff;
            --canvas: #f5f6f8;
            --accent: #0f5f6d;
            --real: #176b52;
            --real-bg: #eef8f4;
            --fake: #a63a32;
            --fake-bg: #fff3f1;
        }

        [data-testid="stAppViewContainer"] {background: var(--canvas);}
        [data-testid="stHeader"] {background: transparent; height: 2.25rem;}
        [data-testid="stToolbar"] {display: none;}
        #MainMenu, footer {visibility: hidden;}

        .block-container {
            max-width: 1180px;
            padding-top: 1rem;
            padding-bottom: 1.25rem;
        }
        h1, h2, h3, p, label {letter-spacing: 0;}

        .app-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 0.15rem 0 0.9rem;
            margin-bottom: 0.9rem;
            border-bottom: 1px solid var(--line);
        }
        .app-kicker {
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            margin-bottom: 0.2rem;
        }
        .app-title {
            color: var(--ink);
            font-size: 1.8rem;
            font-weight: 720;
            line-height: 1.1;
        }
        .app-subtitle {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.3rem;
        }
        .header-meta {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.4rem;
        }
        .meta-item {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 4px;
            color: #3c4651;
            font-size: 0.76rem;
            font-weight: 650;
            padding: 0.3rem 0.48rem;
            white-space: nowrap;
        }
        .section-label {
            color: #47515d;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.06em;
            margin: 0.2rem 0 0.35rem;
        }

        [data-testid="stFileUploader"] section {
            min-height: 0;
            padding: 0.65rem 0.85rem;
            background: var(--surface);
            border: 1px dashed #aeb7c2;
            border-radius: 5px;
        }
        [data-testid="stFileUploader"] section > div {min-height: 0;}
        [data-testid="stFileUploader"] small {font-size: 0.72rem;}
        [data-testid="stFileUploaderDropzone"] button p {font-size: 0;}
        [data-testid="stFileUploaderDropzone"] button p::after {
            content: "Seleccionar";
            font-size: 0.875rem;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span {font-size: 0;}
        [data-testid="stFileUploaderDropzoneInstructions"] span::after {
            content: "MP4 · máximo 500 MB";
            font-size: 0.75rem;
        }

        [data-testid="stVideo"] {
            background: #11161b;
            border: 1px solid #cfd5dc;
            border-radius: 5px;
            overflow: hidden;
        }
        [data-testid="stVideo"] video {
            display: block;
            width: 100%;
            max-height: 390px;
            object-fit: contain;
            background: #11161b;
        }
        .file-caption {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            color: var(--muted);
            font-size: 0.76rem;
            margin-top: 0.35rem;
        }
        .file-caption span:first-child {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .result-panel {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border: 1px solid var(--line);
            border-left-width: 5px;
            border-radius: 5px;
            padding: 0.8rem 0.9rem;
            margin: 0.55rem 0 0.65rem;
        }
        .result-real {border-left-color: var(--real); background: var(--real-bg);}
        .result-fake {border-left-color: var(--fake); background: var(--fake-bg);}
        .result-eyebrow {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 720;
            letter-spacing: 0.05em;
        }
        .result-label {
            color: var(--ink);
            font-size: 1.6rem;
            font-weight: 760;
            line-height: 1.05;
            margin-top: 0.15rem;
        }
        .result-threshold {
            color: var(--muted);
            font-size: 0.76rem;
            text-align: right;
        }

        div[data-testid="stMetric"] {
            min-height: 76px;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 4px;
            padding: 0.55rem 0.65rem;
        }
        div[data-testid="stMetricLabel"] {font-size: 0.73rem;}
        div[data-testid="stMetricValue"] {font-size: 1.25rem;}

        .diagnostic-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 0.65rem 0;
            border: 1px solid var(--line);
            border-radius: 4px;
            background: var(--surface);
        }
        .diagnostic-item {
            min-width: 0;
            padding: 0.5rem 0.6rem;
            border-right: 1px solid var(--line);
        }
        .diagnostic-item:last-child {border-right: 0;}
        .diagnostic-item span {
            display: block;
            color: var(--muted);
            font-size: 0.66rem;
            line-height: 1.2;
        }
        .diagnostic-item strong {
            display: block;
            color: var(--ink);
            font-size: 0.98rem;
            margin-top: 0.12rem;
        }
        .scope-note {
            color: #56616d;
            font-size: 0.75rem;
            line-height: 1.4;
            border-top: 1px solid var(--line);
            margin-top: 0.65rem;
            padding-top: 0.55rem;
        }
        .empty-result {
            display: flex;
            min-height: 210px;
            align-items: center;
            justify-content: center;
            text-align: center;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 5px;
            color: var(--muted);
            margin-top: 0.55rem;
        }
        .empty-result strong {
            display: block;
            color: var(--ink);
            font-size: 0.92rem;
            margin-bottom: 0.2rem;
        }
        [data-testid="stExpander"] {
            background: var(--surface);
            border-color: var(--line);
            border-radius: 4px;
        }
        [data-testid="stAlert"] {padding: 0.55rem 0.7rem;}
        .stButton > button {border-radius: 4px; font-weight: 680;}

        @media (max-width: 760px) {
            .block-container {padding-top: 0.65rem;}
            .app-header {align-items: flex-start; flex-direction: column; gap: 0.65rem;}
            .header-meta {justify-content: flex-start;}
            .app-title {font-size: 1.55rem;}
            [data-testid="stVideo"] video {max-height: 320px;}
        }
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
            <div>
                <div class="result-eyebrow">CLASIFICACIÓN DEL VIDEO</div>
                <div class="result-label">{result.label.upper()}</div>
            </div>
            <div class="result-threshold">Decisión con<br>τ = {result.threshold:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_column, threshold_column, time_column = st.columns(3)
    score_column.metric("Puntaje Fake", f"{result.fake_score * 100:.2f} %")
    threshold_column.metric("Umbral", f"{result.threshold:.3f}")
    time_column.metric("Tiempo", f"{result.elapsed_seconds:.1f} s")

    diagnostics = result.diagnostics
    st.markdown(
        f"""
        <div class="diagnostic-strip">
            <div class="diagnostic-item">
                <span>Rostros válidos</span><strong>{diagnostics.valid_faces}</strong>
            </div>
            <div class="diagnostic-item">
                <span>Posiciones rellenadas</span><strong>{diagnostics.padded_positions}</strong>
            </div>
            <div class="diagnostic-item">
                <span>Entrada del modelo</span><strong>{diagnostics.selected_positions}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if diagnostics.padded_positions > 0:
        st.warning(
            f"Se detectaron {diagnostics.valid_faces} rostros validos y se "
            f"rellenaron {diagnostics.padded_positions} posiciones repitiendo "
            "el ultimo rostro disponible."
        )

    with st.expander("Detalles tecnicos"):
        details_left, details_right = st.columns(2)
        details_left.caption(f"Dispositivo: `{result.device}`")
        details_left.caption(
            f"Fotogramas del video: `{diagnostics.total_video_frames}`"
        )
        details_right.caption(
            f"Posiciones decodificadas: `{diagnostics.decoded_positions}`"
        )
        details_right.caption(
            "Logits Real/Fake: "
            f"`({result.logits[0]:.5f}, {result.logits[1]:.5f})`"
        )

    st.markdown(
        """
        <div class="scope-note">
            <strong>Alcance:</strong> el Puntaje Fake no es una probabilidad
            calibrada. El modelo fue evaluado con FaceForensics++ C23 y su
            resultado no certifica la autenticidad del video.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="app-header">
        <div>
            <div class="app-kicker">PROTOTIPO DE INFERENCIA</div>
            <div class="app-title">Detector de Deepfakes</div>
            <div class="app-subtitle">Clasificación espacial-espectral a nivel de video</div>
        </div>
        <div class="header-meta">
            <span class="meta-item">50 fotogramas</span>
            <span class="meta-item">256 DCT</span>
            <span class="meta-item">τ = 0.333</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("Cargando modelos de inferencia..."):
        resources = get_resources()
except DeepfakeAppError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"No fue posible iniciar los modelos: {exc}")
    st.stop()

st.markdown('<div class="section-label">VIDEO DE ENTRADA</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Video MP4",
    type=["mp4"],
    accept_multiple_files=False,
    max_upload_size=500,
    label_visibility="collapsed",
    width="stretch",
)

if uploaded is None:
    st.markdown(
        """
        <div class="empty-result">
            <div><strong>Sin video cargado</strong>Admite un archivo MP4 de hasta 500 MB.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

video_bytes = uploaded.getvalue()
upload_digest = hashlib.sha256(video_bytes).hexdigest()
if st.session_state.get("upload_digest") != upload_digest:
    st.session_state["upload_digest"] = upload_digest
    st.session_state.pop("inference_result", None)

preview_column, result_column = st.columns([1.42, 1], gap="large")

with preview_column:
    st.markdown('<div class="section-label">VISTA PREVIA</div>', unsafe_allow_html=True)
    st.video(video_bytes)
    safe_name = html.escape(uploaded.name)
    file_size_mb = len(video_bytes) / (1024 * 1024)
    st.markdown(
        f"""
        <div class="file-caption">
            <span>{safe_name}</span><span>{file_size_mb:.1f} MB</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with result_column:
    st.markdown('<div class="section-label">RESULTADO</div>', unsafe_allow_html=True)
    analyze = st.button(
        "Analizar video",
        type="primary",
        icon=":material/search_insights:",
        width="stretch",
    )

    if analyze:
        progress = st.progress(0, text="Preparando análisis")

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
            st.error(f"El análisis no pudo completarse: {exc}")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    stored_result = st.session_state.get("inference_result")
    if stored_result is None:
        st.markdown(
            """
            <div class="empty-result">
                <div><strong>Pendiente de análisis</strong>El resultado aparecerá en este panel.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_result(stored_result)
