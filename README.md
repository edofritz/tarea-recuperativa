## Tarea Cloud Computing

Ciclo completo, desde el entrenamiento a la nube

Docente : **Cristhian Aguilera**

**Este proyecto requiere Docker y Docker compose**

Para levantar el proyecto debe ejecutar el siguiente comando dentro de la raíz del proyecto:

`docker-compose up -d`

Esto levantará 3 contenedores

Una vez levantados los contenedores podemos acceder a la aplicación ingresando al enlace http://localhost:8054

También se debe editar el archivo front/index.html cambiando la ip del servicio backend que en el caso de un ambiente local debe ser http://localhost:8050

Para entrenar el modelo se debe ejecutar:

`docker-compose exec training python train.py`

Este comando generara los nuevos archivos de los modelos.

**Eduardo Peña Fritz**
