FROM python:3.10-slim

WORKDIR /app

# librerías necesarias para la inferencia
COPY requirements.txt .
RUN pip install -r requirements.txt

# script, red entrenada y preprocesamiento
COPY inferencia.py modelo.h5 preprocesamiento.joblib ./

CMD ["python", "inferencia.py"]
