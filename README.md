## Tarea Cloud Computing

Ciclo completo, desde el entrenamiento a la nube

Docente : **Cristhian Aguilera**

**Este proyecto requiere Docker y Docker compose**

---

Para levantar el proyecto se debe ejecutar el siguiente comando dentro de la raíz del proyecto:

`docker-compose up -d`

Esto levantará 4 contenedores

Una vez levantados los contenedores podemos acceder a la aplicación ingresando a

http://localhost:8000

Para re entrenar el modelo debemos:

    - Descomprimir el archivo images.zip reemplazando la carpeta images/
    - Ejecutar el comando:

`docker-compose exec training python train.py`

**Eduardo Peña Fritz**
