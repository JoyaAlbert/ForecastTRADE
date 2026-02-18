# Arquitectura y Características del Modelo Híbrido v2.0

Este documento detalla el funcionamiento interno del pipeline de `ForecastTRADE`, explicando el rol de cada componente del modelo y el propósito de cada una de las características utilizadas.

## Visión General: Arquitectura Híbrida

El modelo combina la capacidad de extracción de secuencias de las redes neuronales recurrentes (LSTM) con la potencia de los árboles de decisión (XGBoost) para la clasificación final.

1.  **LSTM (Feature Extractor):** Aprende representaciones latentes de la dinámica del precio y volumen.
2.  **XGBoost (Classifier):** Toma estas representaciones latentes junto con indicadores técnicos para predecir la probabilidad de éxito de una operación.

---

## Componente 1: La LSTM - Extracción de Características Latentes

### Arquitectura Dual-Task
La LSTM no solo predice el precio, sino que resuelve dos tareas simultáneamente para aprender mejores representaciones:
1.  **Regresión:** Predicción de retornos logarítmicos futuros (minimiza MSE).
2.  **Clasificación:** Predicción de la dirección del movimiento (minimiza Binary Cross-Entropy).

### Latent Space Compression
-   La red extrae un vector de características de 32 dimensiones.
-   Se aplica una **compresión dimensional** a 10 dimensiones para filtrar ruido y quedarse con la "esencia" del movimiento.
-   Estas 10 features latentes (`lstm_latent_0` a `lstm_latent_9`) se pasan al XGBoost.

---

## Componente 2: Triple Barrier Method (Labeling)

Para evitar el ruido de las etiquetas simples (Subida/Bajada), utilizamos el **Triple Barrier Method** adaptativo basado en la volatilidad.

### Definición de Etiquetas
Para cada punto `t`, establecemos tres barreras basadas en la volatilidad local (`σ`):
1.  **Profit Taking (Barrera Superior):** `Precio[t] * (1 + 2.5 * σ)`
2.  **Stop Loss (Barrera Inferior):** `Precio[t] * (1 - 1.25 * σ)`
3.  **Time Horizon (Barrera Vertical):** 20 días de trading.

### Clases
-   **BUY (1):** El precio toca la barrera superior antes que la inferior o el límite de tiempo.
-   **SELL (0):** El precio toca la barrera inferior antes que la superior o el límite de tiempo.
-   **HOLD:** El precio no toca ninguna barrera en 20 días (se descartan estas muestras para entrenamiento para enfocar el modelo en movimientos claros).

---

## Componente 3: XGBoost y Validación Cruzada

### Validation Strategy: Sliding Window
Para garantizar la estabilidad y evitar *data leakage* y sobreajuste a regímenes pasados, utilizamos **Sliding Window Cross-Validation**:
-   **Training Window:** 400 días (fijo, se mueve hacia adelante).
-   **Validation Window:** 120 días (fijo, no solapado).
-   **Embargo:** 20 días (buffer de seguridad entre train y test para evitar leakage por correlación serial).
-   **Folds:** 12 iteraciones ("pasos") que cubren diferentes condiciones de mercado.

### Feature Engineering
El XGBoost recibe un conjunto rico de datos:
-   **Latent Features de LSTM:** 10 variables comprimidas.
-   **Momentum:** RSI, MACD, ROC.
-   **Volatilidad:** ATR, Volatility de 20 días.
-   **Tendencia:** EMAs, Distancia a MA200.
-   **Macro:** Ratio Precio/VIX.

### Selección de Características
El sistema aplica filtros automáticos para eliminar ruido:
-   Eliminación de features altamente correlacionadas (>0.98).
-   Eliminación de features con importancia < 1% en iteraciones previas.

---

## Gestión de Riesgo Dinámica

El modelo no solo predice dirección, sino que sugiere parámetros de gestión de riesgo:
-   **Entry:** Precio de cierre actual.
-   **Profit Target:** Dinámico, basado en la volatilidad proyectada (2.5x desviaciones).
-   **Stop Loss:** Dinámico, ajustado a la volatilidad reciente (1.25x desviaciones).
-   **Win Probability:** Confianza del modelo (soft probability del XGBoost).

---

## Métricas de Éxito
Nos enfocamos en la estabilidad y rentabilidad ajustada al riesgo:
-   **Sharpe Ratio:** Retorno por unidad de riesgo (objetivo principal).
-   **Win Rate:** % de operaciones ganadoras (buscamos > 55%).
-   **Stability:** Varianza baja entre los resultados de los diferentes folds de validación.
