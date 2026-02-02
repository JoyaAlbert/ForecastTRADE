# ForecastTRADE: Hybrid Financial Forecasting Model

Este proyecto implementa un modelo de pronóstico financiero híbrido que combina un modelo de Deep Learning (LSTM) con un modelo de Gradient Boosting (XGBoost) para predecir la dirección del precio de las acciones (en este caso, AAPL).

La arquitectura sigue una filosofía de "equipo de especialistas":
- **LSTM (Analista de Patrones):** Actúa como un experto en análisis técnico visual, procesando secuencias de precios (OHLCV) para detectar inercias y patrones de tendencia.
- **XGBoost (Director de Decisiones):** Recibe la predicción de tendencia de la LSTM como un "indicador experto" y la cruza con otros datos (indicadores técnicos, sentimiento del mercado y volatilidad) para tomar una decisión final sobre si el retorno del día siguiente será positivo.

## Requisitos

- Python 3.8+
- `venv` para la gestión de entornos virtuales

## Cómo Empezar

Siga estos pasos para configurar y ejecutar el proyecto.

### 1. Clonar el Repositorio

Si ha descargado los archivos manualmente, puede omitir este paso.
```bash
git clone <URL_DEL_REPOSITORIO>
cd ForecastTRADE
```

### 2. Crear y Activar el Entorno Virtual

Es una buena práctica aislar las dependencias del proyecto.

```bash
# Crear el entorno virtual
python3 -m venv venv

# Activar el entorno (en Linux/macOS)
source venv/bin/activate
```
*Nota: En Windows, la activación se hace con `.\venv\Scripts\activate`.*

### 3. Instalar las Dependencias

Instale todas las librerías necesarias con el siguiente comando:
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el Pipeline

Para iniciar el proceso de entrenamiento y evaluación, simplemente ejecute el script principal:
```bash
python main.py
```

## ¿Qué esperar de la ejecución?

El script realizará las siguientes acciones:
1.  **Descargará** los datos históricos de precios para AAPL y el índice VIX.
2.  **Entrenará el modelo LSTM** para aprender los patrones de secuencia de precios. Verá el resumen de la arquitectura de la red en la consola.
3.  **Generará un gráfico** `lstm_prediction_vs_real_AAPL.png` que muestra qué tan bien la LSTM predijo los precios en el conjunto de prueba.
4.  **Integrará** la predicción de la LSTM como una nueva característica.
5.  **Entrenará y evaluará el modelo XGBoost** final utilizando validación cruzada de series temporales.
6.  **Imprimirá en la consola** un resumen del rendimiento del modelo (AUC y Accuracy) para cada fold de la validación.
7.  **Mostrará un ranking** de las 10 características más importantes para el modelo XGBoost, permitiéndole ver qué información valora más el "Director de Decisiones".

Al finalizar, habrá ejecutado un pipeline completo de finanzas cuantitativas, desde la ingeniería de características hasta la evaluación de un modelo híbrido complejo.