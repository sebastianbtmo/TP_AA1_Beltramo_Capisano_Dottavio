# predice si mañana llueve a partir de los datos de un día (mismo formato que weatherAUS)
import numpy as np
import pandas as pd
import joblib
from sklearn.experimental import enable_iterative_imputer  # necesario para cargar el IterativeImputer
import tensorflow as tf

# red y preprocesamiento guardados desde el notebook
modelo = tf.keras.models.load_model("modelo.h5", compile=False)
prep = joblib.load("preprocesamiento.joblib")


def preparar(df):
    # columnas que no se usan para predecir (si vienen en el archivo)
    df = df.drop(columns=["Unnamed: 0", "RainfallTomorrow", "RainTomorrow"], errors="ignore")

    # mes cíclico con seno y coseno, y RainToday a 0/1
    mes = pd.to_datetime(df["Date"]).dt.month
    df["Month_sin"] = np.sin(2 * np.pi * mes / 12)
    df["Month_cos"] = np.cos(2 * np.pi * mes / 12)
    df["RainToday"] = df["RainToday"].map({"No": 0, "Yes": 1})
    df = df.drop(columns=prep["cols_a_eliminar"] + ["Date"], errors="ignore")

    # la ciudad se reemplaza por su latitud y longitud
    df["Lat"] = df["Location"].map(lambda c: prep["coordenadas"][c][0])
    df["Lon"] = df["Location"].map(lambda c: prep["coordenadas"][c][1])
    return df.drop(columns="Location")


def imputar(X):
    X = X.copy()
    # sin viento no hay dirección: se marca como "Calma"
    for dir_col, vel_col in [("WindDir9am", "WindSpeed9am"), ("WindDir3pm", "WindSpeed3pm")]:
        X.loc[X[dir_col].isna() & (X[vel_col] == 0), dir_col] = "Calma"
    # numéricas con el imputer de train y categóricas con la moda de train
    cols_num = prep["cols_num"]
    X[cols_num] = prep["imputer_num"].transform(X[prep["cols_imp"]])[:, :len(cols_num)]
    X[prep["cols_cat"]] = X[prep["cols_cat"]].fillna(prep["modas"])
    return X


def codificar(X):
    X = X.copy()
    # direcciones de viento a seno y coseno de su ángulo ("Calma" queda en 0, 0)
    for c in prep["cols_viento"]:
        ang = X[c].map(prep["angulo"])
        X[c + "_sin"] = np.sin(ang).fillna(0)
        X[c + "_cos"] = np.cos(ang).fillna(0)
    return X.drop(columns=prep["cols_viento"])


def escalar(X):
    X = X.copy()
    # se escala con la media y el desvío de train
    X[prep["cols_num"]] = prep["scaler"].transform(X[prep["cols_num"]])
    return X


# datos del día, sin las ciudades que no estaban en el entrenamiento
datos = pd.read_csv("datos/entrada.csv")
datos = datos[datos["Location"].isin(list(prep["coordenadas"]))]

# mismos pasos y mismo orden de columnas que en el entrenamiento
X = escalar(codificar(imputar(preparar(datos))))[prep["columnas_finales"]]
proba = modelo.predict(X.to_numpy("float32"), verbose=0).ravel()

# se guarda la probabilidad y la predicción con el umbral elegido en validación
resultado = pd.DataFrame({
    "Date": datos["Date"].values,
    "Location": datos["Location"].values,
    "probabilidad_lluvia": proba.round(4),
    "llueve_manana": np.where(proba >= prep["umbral"], "Yes", "No"),
})
resultado.to_csv("datos/predicciones.csv", index=False)
print(resultado.to_string(index=False))
