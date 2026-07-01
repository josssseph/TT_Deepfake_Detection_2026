# Detector de Deepfakes con Docker

Interfaz local para clasificar un video MP4 como `Real` o `Fake` mediante el modelo espacial-espectral desarrollado en el proyecto de titulacion. Docker encapsula Python, PyTorch, OpenCV, Streamlit, los pesos y la configuracion necesaria para ejecutar la inferencia sin instalar dependencias en el equipo anfitrion.

## Configuracion congelada

| Componente | Valor |
|---|---:|
| Checkpoint | `ft_spatial_spectral_bs64_lr1e-5_wd1e-3.pth` |
| Fotogramas base | 60 |
| Fotogramas del modelo | 50 |
| Coeficientes DCT | 256 |
| Backbone | ResNet-18 `IMAGENET1K_V1` |
| Modalidades | Espacial y espectral |
| Umbral Fake | 0.333 |

El valor mostrado por la interfaz es un **Puntaje Fake**, no una probabilidad calibrada.

## Requisitos

- Docker Desktop o Docker Engine con contenedores Linux.
- Docker Compose v2 o posterior.
- Arquitectura AMD64.
- Aproximadamente 4 GB de RAM disponibles y 3 GB de espacio libre.
- Internet durante la primera construccion de la imagen.

No se requiere Python, PyTorch, FFmpeg ni GPU en el equipo anfitrion.

## Inicio rapido

Desde la raiz del repositorio:

```bash
docker compose up --build
```

Abra en el navegador:

```text
http://localhost:8501
```

Seleccione un video MP4, revise la vista previa y pulse **Analizar video**. La primera version procesa un archivo por vez y admite hasta 500 MB.

Para ejecutar en segundo plano:

```bash
docker compose up --build --detach
docker compose logs --follow
```

Para detener y retirar el contenedor:

```bash
docker compose down
```

El puerto puede cambiarse mediante una variable de entorno:

```bash
APP_PORT=8600 docker compose up --build
```

En PowerShell:

```powershell
$env:APP_PORT=8600
docker compose up --build
```

## Verificaciones

El `Dockerfile` realiza las siguientes comprobaciones durante la construccion:

1. Instala versiones fijadas de las dependencias.
2. Descarga y almacena ResNet-18 `IMAGENET1K_V1` dentro de la imagen.
3. Verifica mediante SHA-256 el checkpoint y los dos archivos del detector facial.
4. Carga estrictamente el checkpoint en la arquitectura de 50 fotogramas y 256 DCT.
5. Ejecuta las pruebas automaticas.

Para repetir las pruebas dentro de una imagen construida:

```bash
docker compose run --rm app python -m pytest -q -p no:cacheprovider
```

Para comprobar el estado del servicio:

```bash
docker compose ps
```

## Privacidad y seguridad

- El contenedor se ejecuta como usuario sin privilegios.
- El sistema de archivos se monta como solo lectura.
- Los videos se procesan en `/tmp`, que es temporal y se elimina junto con el contenedor.
- No se envian videos ni resultados a servicios externos durante la inferencia.
- ResNet se descarga durante la construccion; la ejecucion posterior puede realizarse sin Internet.

## Reproducibilidad

- Imagen base fijada a Python 3.11.9 sobre Debian Bookworm mediante digest para Linux/AMD64.
- Dependencias transitivas registradas en `requirements.lock`.
- PyTorch 2.5.1 y Torchvision 0.20.1 instalados desde el indice oficial para CPU.
- Configuracion e integridad de los modelos conservadas en `model_manifest.json`.
- Nombre de imagen local: `deepfake-detector:1.0.1`.

Cambiar un recurso requiere actualizar su hash en el manifiesto y volver a construir la imagen. No debe ignorarse una discrepancia de integridad.

## Estructura

```text
deepfake-detector-docker/
|-- app.py
|-- deepfake_app/
|-- scripts/model.py
|-- assets/face_detector/
|-- models/finetuned/
|-- tests/
|-- model_manifest.json
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.lock
`-- README.md
```

## Limitaciones

El modelo fue evaluado con FaceForensics++ C23 y las manipulaciones Deepfakes y Face2Face. Su clasificacion no certifica la autenticidad de un video y puede no generalizar a otros metodos, niveles de compresion, ediciones o condiciones de captura.

La validacion de paridad con videos de referencia debe completarse antes de presentar la interfaz como evidencia experimental adicional. Los videos de validacion deben mantenerse fuera del repositorio salvo que sus condiciones de uso permitan redistribuirlos.

## Publicacion del repositorio

Revise primero `NOTICE.md` y defina con los autores la licencia del codigo y los permisos de redistribucion de los pesos. Despues:

```bash
git init
git add .
git commit -m "Agregar interfaz reproducible con Docker"
git branch -M main
git remote add origin URL_DEL_REPOSITORIO
git push -u origin main
```
