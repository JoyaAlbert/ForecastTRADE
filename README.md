# 📈 ForecastTRADE v2.0

**Hybrid LSTM-XGBoost Stock Prediction System with Volatility-Adaptive Risk Management**

ForecastTRADE es un sistema de trading algorítmico avanzado que combina el poder del Deep Learning (LSTM) para la extracción de características con la precisión de los árboles de decisión (XGBoost) para la ejecución táctica. Está diseñado para operar con estabilidad y robustez en múltiples regímenes de mercado.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)

---

## 🚀 Características Clave (v2.0)

-   **🧠 Arquitectura Híbrida:** 
    -   **LSTM (Dual-Task):** Aprende la "física" del mercado resolviendo regresión (precio) y clasificación (dirección) simultáneamente.
    -   **XGBoost (Ensemble):** Toma decisiones finales basadas en el estado latente de la LSTM y señales técnicas.
-   **🛡️ Triple Barrier Method:** Etiquetado inteligente de datos basado en volatilidad local (TP=2.5σ, SL=1.25σ) para evitar ruido.
-   **📅 Sliding Window Validation:** Estrategia de validación cruzada (12 folds) con embargo period (20d) para eliminar *data leakage*.
-   **📌 Estrategia explícita `long_only`:** El motor de trading abre solo posiciones largas; métricas y umbrales alineados con esa política.
-   **🧾 Feature Contract:** El pipeline resuelve un contrato final de features para evitar inconsistencias entre seed features y filtros.
-   **⚖️ Gestión de Riesgo Dinámica:** Sugiere volumen de posición y niveles de Stop-Loss adaptados a la volatilidad del mercado.
-   **✨ Rich UI:** Interfaz de terminal profesional con barras de progreso, tablas y gráficos ASCII.

---

## 🛠️ Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd ForecastTRADE
    ```

2.  **Crear entorno virtual:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Uso

El punto de entrada principal es `run.py`, que gestiona todo el ciclo de vida de la predicción.

```bash
# Asegúrate de tener el entorno activado
source .venv/bin/activate

# Ejecutar el sistema
python run.py
```

### Flujo de Ejecución:
1.  **Selección de Activo:** Elija un ticker (ej. MSFT, NVDA, AAPL) desde el menú interactivo.
2.  **Data Fetching:** Se descargan datos históricos de Yahoo Finance.
3.  **LSTM Training:** Se entrena la red neuronal para extraer "Latent Features" (representaciones comprimidas del mercado).
4.  **Feature Engineering:** Se calculan indicadores técnicos avanzados y se filtran por importancia.
5.  **Backtesting (Folds Configurados/Validos):** Se ejecuta validación deslizante con tamaño mínimo de validación para excluir folds no comparables.
6.  **Final Recommendation:** Se genera una señal de trading (BUY/SELL/HOLD) con niveles de precio específicos.

---

## 📊 Interpretación de Resultados

Al finalizar, el sistema generará:
-   **Gráficos en `out/`:** Visualizaciones de las predicciones vs realidad para cada fold.
-   **Resumen de Consola:**
    -   **Win Probability:** Probabilidad estimada de éxito.
    -   **Recommendation:** Acción sugerida (requiere >65% para BUY).
    -   **Dynamic Risk:** Niveles de Profit Target y Stop Loss calculados dinámicamente.

---

## 🏗️ Arquitectura

Para detalles técnicos profundos sobre cómo funciona el Dual-Task LSTM, el Triple Barrier Method y la ingeniería de características, consulte [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).
