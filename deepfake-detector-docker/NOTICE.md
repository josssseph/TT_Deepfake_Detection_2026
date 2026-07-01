# Aviso sobre codigo y modelos

Este repositorio combina codigo desarrollado para el proyecto de titulacion con recursos de terceros y pesos obtenidos durante el experimento.

- El checkpoint `ft_spatial_spectral_bs64_lr1e-5_wd1e-3.pth` corresponde al modelo seleccionado para la interfaz.
- Su historial de validacion se conserva en `models/finetuned/ft_spatial_spectral_bs64_lr1e-5_wd1e-3_history.csv`.
- ResNet-18 utiliza los pesos `IMAGENET1K_V1` distribuidos mediante Torchvision.
- La deteccion facial utiliza la arquitectura SSD Caffe consumida por OpenCV DNN.
- Los videos de FaceForensics++ no se incluyen en este repositorio.

Antes de publicar o redistribuir este repositorio, el responsable debe comprobar las condiciones aplicables al checkpoint, a los pesos preentrenados, al detector facial y a cualquier material de validacion. No se ha asignado automaticamente una licencia global porque esa decision corresponde a los autores del proyecto.
