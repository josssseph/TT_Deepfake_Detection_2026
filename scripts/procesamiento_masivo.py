import cv2
import os
import h5py
import numpy as np
from tqdm import tqdm
from scipy.fftpack import dct

# ============================================================
# 1. CONFIGURACIÓN DE RUTAS ABSOLUTAS
# ============================================================
USER_HOME = '/home/joseph.jaramillo__ucuenca.edu.ec/deepfake_project'
dataset_path = f'{USER_HOME}/data/raw/FaceForensics++_C23'
output_h5 = f'{USER_HOME}/data/processed/ff_dataset_max60frames_4096dct.h5'  # NUEVO NOMBRE

# Modelos DNN
prototxt_path = f"{USER_HOME}/scripts/deploy.prototxt"
caffemodel_path = f"{USER_HOME}/scripts/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# ============================================================
# PARÁMETROS MÁXIMOS (se pueden reducir después en el Dataset)
# ============================================================
MAX_FRAMES = 60                # Máximo de frames que vamos a guardar
N_DCT_COEFFS_MAX = 4096        # Máximo de coeficientes DCT por frame

# ============================================================
# 2. FUNCIONES AUXILIARES (sin cambios salvo la DCT)
# ============================================================

def extraer_video_id(nombre_archivo, carpeta):
    """
    Extrae el identificador del video original (0-999) a partir del nombre del archivo.
    - Para 'original': '000.mp4' → 0
    - Para 'Deepfakes', 'Face2Face', etc.: '000_003.mp4' → 0 (primer número antes del guión bajo)
    """
    nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
    if carpeta == 'original':
        return int(nombre_sin_ext)
    else:
        partes = nombre_sin_ext.split('_')
        return int(partes[0]) if len(partes) >= 1 else -1

def zigzag_indices(n):
    """
    Genera índices en orden zigzag para una matriz cuadrada n x n.
    Retorna lista de tuplas (fila, columna).
    """
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            i = min(s, n - 1)
            j = s - i
            while i >= 0 and j < n:
                indices.append((i, j))
                i -= 1
                j += 1
        else:
            j = min(s, n - 1)
            i = s - j
            while j >= 0 and i < n:
                indices.append((i, j))
                i += 1
                j -= 1
    return indices

# Precalcular índices zigzag para imagen 224x224 (hasta el máximo que necesitemos)
ZIGZAG_224 = zigzag_indices(224)

def extraer_dct_frame(rostro_gray, num_coeffs):
    """
    Calcula la DCT 2D de la imagen en escala de grises (224x224),
    y devuelve los primeros num_coeffs coeficientes en orden zigzag.
    
    Args:
        rostro_gray: numpy array (224, 224), valores float en [0,1].
        num_coeffs: número de coeficientes a conservar.
    Returns:
        numpy array de longitud num_coeffs.
    """
    # Aplicar DCT 2D (norm='ortho' para mantener energía)
    dct_frame = dct(dct(rostro_gray.T, norm='ortho').T, norm='ortho')
    # Recorrer en zigzag y tomar los primeros num_coeffs
    coefs = np.array([dct_frame[i, j] for i, j in ZIGZAG_224[:num_coeffs]])
    return coefs.astype('float32')

# ============================================================
# 3. FUNCIÓN DE PROCESAMIENTO (1 Vídeo → 60 frames + DCTs)
# ============================================================
def extraer_secuencia_rostros(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Muestreo uniforme de MAX_FRAMES índices (aunque total_frames sea menor)
    indices_muestreo = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int)
    
    secuencia_rostros = []
    secuencia_dct = []
    
    for idx in indices_muestreo:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        exito, frame = cap.read()
        if not exito:
            continue
            
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        
        confianza = detections[0, 0, :, 2]
        mejor_idx = np.argmax(confianza)
        
        if confianza[mejor_idx] > 0.5:
            box = detections[0, 0, mejor_idx, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            rostro = frame[y1:y2, x1:x2]
            if rostro.size != 0:
                # Redimensionar y normalizar (se guarda en BGR, igual que antes)
                rostro_224 = cv2.resize(rostro, (224, 224), interpolation=cv2.INTER_AREA)
                rostro_norm = rostro_224.astype("float32") / 255.0
                
                # Convertir a escala de grises para DCT
                rostro_gray = (0.299 * rostro_norm[:,:,2] + 
                               0.587 * rostro_norm[:,:,1] + 
                               0.114 * rostro_norm[:,:,0])  # BGR -> Gris
                
                # Extraer los coeficientes DCT (hasta N_DCT_COEFFS_MAX)
                dct_features = extraer_dct_frame(rostro_gray, N_DCT_COEFFS_MAX)
                
                secuencia_rostros.append(rostro_norm)
                secuencia_dct.append(dct_features)
    
    cap.release()
    
    # -----------------------------------------------
    # 4. POSTPROCESADO: PADDING HASTA MAX_FRAMES
    # -----------------------------------------------
    if len(secuencia_rostros) == 0:
        # Si no se detectó ningún rostro en toda la secuencia, descartamos
        return None, None
    
    # Si tenemos menos de MAX_FRAMES, rellenamos con el último rostro válido
    while len(secuencia_rostros) < MAX_FRAMES:
        secuencia_rostros.append(secuencia_rostros[-1])
        secuencia_dct.append(secuencia_dct[-1])
    
    # Si por alguna razón tenemos más de MAX_FRAMES (no debería), truncamos
    secuencia_rostros = secuencia_rostros[:MAX_FRAMES]
    secuencia_dct = secuencia_dct[:MAX_FRAMES]
    
    return (np.array(secuencia_rostros),   # (60, 224, 224, 3)
            np.array(secuencia_dct))       # (60, 4096)

# ============================================================
# 5. ESCRITURA PROGRESIVA EN HDF5
# ============================================================
with h5py.File(output_h5, 'w') as h5f:
    # Datasets con las dimensiones máximas
    dset_x = h5f.create_dataset('X', 
                                shape=(0, MAX_FRAMES, 224, 224, 3), 
                                maxshape=(None, MAX_FRAMES, 224, 224, 3), 
                                dtype='float32', chunks=True)
    dset_y = h5f.create_dataset('Y', shape=(0,), maxshape=(None,), dtype='int8')
    dset_id = h5f.create_dataset('video_id', shape=(0,), maxshape=(None,), dtype='int16')
    
    dset_dct = h5f.create_dataset('X_dct',
                                  shape=(0, MAX_FRAMES, N_DCT_COEFFS_MAX),
                                  maxshape=(None, MAX_FRAMES, N_DCT_COEFFS_MAX),
                                  dtype='float32', chunks=True)
    
    def procesar_carpeta(carpeta, etiqueta):
        directorio = os.path.join(dataset_path, carpeta)
        archivos = os.listdir(directorio)
        
        for video in tqdm(archivos, desc=f"Procesando {carpeta}"):
            if not video.endswith('.mp4'):
                continue
            
            ruta_completa = os.path.join(directorio, video)
            tensor_video, tensor_dct = extraer_secuencia_rostros(ruta_completa)
            
            if tensor_video is not None:   # solo si se detectó al menos un rostro
                vid = extraer_video_id(video, carpeta)
                curr_size = dset_x.shape[0]
                
                # Agrandar todos los datasets en 1
                dset_x.resize((curr_size + 1, MAX_FRAMES, 224, 224, 3))
                dset_y.resize((curr_size + 1,))
                dset_id.resize((curr_size + 1,))
                dset_dct.resize((curr_size + 1, MAX_FRAMES, N_DCT_COEFFS_MAX))
                
                # Escribir
                dset_x[curr_size] = tensor_video
                dset_y[curr_size] = etiqueta
                dset_id[curr_size] = vid
                dset_dct[curr_size] = tensor_dct

    print("Iniciando procesamiento masivo...")
    procesar_carpeta('original', etiqueta=0)
    procesar_carpeta('Deepfakes', etiqueta=1)
    procesar_carpeta('Face2Face', etiqueta=1)

print("\nProcesamiento Masivo Completado")
