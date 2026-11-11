# TP_AA1_Beltramo_Capisano_Dottavio

## Integrantes:
- Sebastián Beltramo
- Martin Capisano
- Lisandro Dottavio

Realización del trabajo práctico número 1 y 2 de la materia Aprendizaje Automático de la tecnicatura universitaria en inteligencia artificial.\
El trabajo 1 consta del análisis del dataset "house-prices.csv" para poder predecir los precios de las casas mediante el uso de modelos de regresión lineal.\
El trabajo 2 consta del análisis del dataset "weatherAUS_2026C2.csv" para poder predecir a las 23:59 de cada día, si al día siguiente va a llover, es un problema de clasificación binaria.
Para llegar a nuestro objetivo hacemos uso de la librería scikit-learn y descenso_gradiente.py para los modelos de regresión lineal, además utilizamos pandas y numpy para el manejo del dataset y matplotlib y seaborn para los gráficos.\
Para el segundo trabajo también hacemos uso de librerias como shap o pycaret.\

## Cómo ejecutar el proyecto

### Requisitos
- Python **3.10.6**, versión utilizada durante el desarrollo del trabajo.
- `pip`, el gestor de paquetes de Python.
- El archivo `requirements.txt` incluido en el repositorio.

Se recomienda utilizar Python 3.10.6 para evitar posibles incompatibilidades entre las dependencias. El proyecto fue desarrollado con esa versión y no se garantiza su correcto funcionamiento con otras versiones de Python o con versiones diferentes de las librerías especificadas.

### Instalación

1. Clonar el repositorio y ubicarse en la carpeta del proyecto.
2. Crear un entorno virtual:

   ```bash
   py -3.10 -m venv venv
   ```

3. Activar el entorno virtual en Windows:

   ```powershell
   .\venv\Scripts\activate
   ```

   En Linux o macOS:

   ```bash
   source venv/bin/activate
   ```

4. Instalar las dependencias:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

### Ejecución

Abrir el notebook del trabajo asegurándose de seleccionar el entorno virtual creado como kernel.