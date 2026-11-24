# Docker - predicción de lluvia

Predice si mañana llueve con la red neuronal del TP, a partir de los datos de un día.

## Requisitos

Tener Docker instalado (en Windows, con Docker Desktop abierto). Todos los comandos se corren desde esta carpeta.

## Pasos

1. Construir la imagen:

```
docker build -t lluvia-aus .
```

2. Poner en esta carpeta un archivo `entrada.csv` con los días a predecir. Tiene que tener las mismas columnas que `weatherAUS_2026C2.csv` (se pueden copiar algunas filas de ese archivo). `RainTomorrow` no hace falta.

3. Correr el contenedor:

```
docker run --rm -v "${PWD}:/app/datos" lluvia-aus
```

Así es en PowerShell. En cmd va `%cd%` en lugar de `${PWD}`, y en Linux o Mac `$(pwd)`.

## Resultado

Se genera `predicciones.csv` en esta carpeta, con la fecha, la ciudad, la probabilidad de lluvia y si llueve o no. Se predice que llueve cuando la probabilidad es 0.327 o más, el umbral que elegimos en el notebook.
