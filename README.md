# Sistema de detección de Deepfakes mediante técnicas de procesamiento de imágenes y aprendizaje automático

Trabajo de titulación desarrollado en la Universidad de Cuenca para construir, entrenar y evaluar un sistema de detección de Deepfakes sobre videos faciales. El proyecto combina procesamiento de imágenes, análisis espectral, métricas temporales y aprendizaje profundo para clasificar secuencias de video como `Real` o `Fake`.

## Autores y tutor

- Carlos Adolfo Calle García
- Joseph Mateo Jaramillo Gómez
- Tutor: Santiago Renán González Martínez

## Propósito del proyecto

La idea central es detectar manipulaciones faciales en video usando una arquitectura modular. El flujo del trabajo parte de videos crudos, extrae rostros y características en distintos dominios, organiza los datos en archivos HDF5, prueba varias configuraciones del modelo y termina con una evaluación final sobre el conjunto de prueba.

El sistema fue desarrollado en un HPC con recursos compartidos, por lo que muchos scripts usan rutas absolutas, trabajos `sbatch` y dependencias fijadas para evitar fallos de reproducibilidad.

## Resumen técnico del enfoque

El trabajo usa tres tipos de información:

- Información espacial, obtenida con una red ResNet-18 congelada.
- Información espectral, obtenida con la Transformada Discreta del Coseno (DCT).
- Información temporal y de consistencia, obtenida con SSIM y `jitter`.

En la fase experimental se trabajó con FaceForensics++ bajo compresión media C23 y secuencias de 50 fotogramas por video. La tesis mostró que la combinación espacial-espectral truncada a 256 coeficientes fue la configuración más estable, mientras que la tercera rama con métricas escalares no aportó mejora suficiente por el desbalance de dimensionalidad.

## Estructura del repositorio

- `scripts/`: contiene el flujo completo de preprocesamiento, precomputación, entrenamiento, ajuste fino y evaluación.
- `deepfake-detector-docker/`: contiene la interfaz gráfica contenedorizada para la demostración final.
- `models/`: guarda pesos y registros del entrenamiento.
- `resultados_finales/`: guarda el resumen de métricas finales.

Para la parte de la GUI y la ejecución con Docker, existe un archivo específico dentro de `deepfake-detector-docker/README.md` con la configuración congelada de la demostración.

## Requisitos generales en HPC

El proyecto fue desarrollado en un entorno compartido, por lo que el flujo esperado es similar al siguiente:

```bash
module load python/3.11
python -m venv ~/deepfake_project/env_df
source ~/deepfake_project/env_df/bin/activate
pip install --upgrade pip
pip install kaggle
pip install "numpy<2" h5py opencv-python-headless tqdm --only-binary :all:
pip install scipy==1.13.1 --only-binary scipy
pip install scikit-learn --only-binary :all:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics --only-binary :all:
pip install seaborn --only-binary :all:
pip install matplotlib --only-binary :all:
```

Notas útiles:

- Se recomienda usar un entorno virtual separado para este proyecto.
- PyTorch y TorchVision se instalaron con binarios oficiales para CUDA 12.1.
- Algunas dependencias, como `scipy`, pueden requerir una versión concreta según el nodo del HPC.
- Para más detalle revisar `requirements.txt`

## Flujo general de uso

El orden recomendado es este:

1. Preprocesar los videos y construir el HDF5 base.
2. Generar las particiones de entrenamiento, validación y prueba.
3. Cálculo de características espaciales y temporales.
4. Alinear las métricas temporales con el muestreo usado por el entrenamiento.
5. Ejecutar el barrido de hiperparámetros.
6. Hacer el ajuste fino de los mejores modelos.
7. Evaluar el mejor modelo sobre el conjunto de prueba.
8. Si se desea una demostración visual, usar la GUI contenedorizada en Docker.

## Guía de scripts

### 1. Preprocesamiento masivo

Script principal: `scripts/procesamiento_masivo.py`

SLURM: `scripts/lanzar_procesamiento.sh`

Función:

- Lee videos de FaceForensics++ C23.
- Detecta rostros con el detector facial basado en OpenCV DNN.
- Extrae hasta 60 fotogramas por video.
- Genera la representación frecuencial con DCT.
- Guarda un HDF5 base con imágenes, DCT, etiquetas e identificador del video.

Uso típico en HPC:

```bash
sbatch scripts/lanzar_procesamiento.sh
```

### 2. División de datos

Script principal: `scripts/split_data.py`

SLURM: `scripts/run_split_data.sh`

Función:

- Crea particiones separadas para entrenamiento, validación y prueba.
- Evita mezcla de identidades o videos entre subconjuntos.
- Guarda índices `.npy` para que el entrenamiento sea reproducible.

### 3. Cálculo de características

Script principal: `scripts/precompute_features.py`

SLURM: `scripts/lanzar_precompute.sh`

Función:

- Lee el HDF5 base.
- Calcula características espaciales con ResNet-18 congelada.
- Calcula SSIM y `jitter`.
- Genera un HDF5 más liviano para entrenamiento.

Este paso suele ejecutarse en GPU porque acelera la extracción de rasgos.

### 4. Alineación temporal

Script principal: `scripts/precompute_temporal_aligned.py`

SLURM: `scripts/lanzar_temporal_aligned.sh`

Función:

- Recalcula las métricas temporales para que coincidan con el muestreo real de cada longitud de secuencia.
- Mantiene alineadas las modalidades usadas por el modelo.
- Organiza la salida en una estructura temporal por número de fotogramas.

### 5. Entrenamiento y tuning inicial

Script principal: `scripts/train_tuning.py`

SLURM: `scripts/run_experiments.sh`

Función:

- Entrena configuraciones candidatas sobre las características precomputadas.
- Permite probar variantes con y sin ramas espaciales, espectrales o temporales.
- Usa métricas de validación para seleccionar candidatos.

### 6. Tuning V2

Script principal: `scripts/train_tuning_v2.py`

SLURM: `scripts/run_tuning_v2.sh`

Función
:

- Ejecuta una búsqueda más ordenada y reproducible.
- Fija semillas y registra el historial de entrenamiento.
- Guarda puntos de control del mejor `validation loss` y del mejor `AUC`.

### 7. Fine-tuning de mejores modelos

Script principal: `scripts/finetuning_expert.py`

SLURM: `scripts/run_finetuning_grid.sh`

Función:

- Toma los modelos mejor posicionados del tuning.
- Aplica una tasa de aprendizaje más baja y regularización más fuerte.
- Reentrena con un barrido pequeño de hiperparámetros para afinar el campeón.

### 8. Evaluación final

Script principal: `scripts/evaluar_test.py`

SLURM: `scripts/test_final.sh`

Función:

- Carga el mejor modelo final.
- Evalua solo el conjunto de prueba.
- Genera reporte de clasificación, matriz de confusión y curva ROC.

## Orden recomendado de ejecución

```text
1. lanzar_procesamiento.sh
2. run_split_data.sh
3. lanzar_precompute.sh
4. lanzar_temporal_aligned.sh
5. run_experiments.sh o run_tuning_v2.sh
6. run_finetuning_grid.sh
7. test_final.sh
```

## Resultados reportados en la tesis

El modelo final reportó, sobre el conjunto de prueba, un `Recall` de 84.00%, un `F1-Score` de 0.8528 y un `AUC` de 0.8654. En el análisis comparativo, la mejor configuración fue la fusión espacial-espectral con 256 coeficientes DCT, ya que la tercera rama con métricas escalares no mejoró el rendimiento de forma suficiente debido a la desproporción de las dimensiones de los vectores característicos.

## GUI en Docker

La interfaz gráfica usada para la demostración final está contenedorizada en `deepfake-detector-docker/`. En caso de necesitar reproducir solo la presentación visual del modelo, revisa ese subdirectorio y su README propio, donde se documenta la ejecución con Docker y la configuración congelada de la inferencia.

## Recomendación final

Este repositorio mezcla investigación, preprocesamiento, entrenamiento y demostración. Si va reutilizarse, empiezar por la lectura de este README, después revisar `scripts/` en el orden de ejecución y al final consulta `deepfake-detector-docker/README.md` si se busca la GUI de la evidencia final.
