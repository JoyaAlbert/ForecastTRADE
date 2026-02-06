# Arquitectura y Características del Modelo Híbrido

Este documento detalla el funcionamiento interno del pipeline de `ForecastTRADE`, explicando el rol de cada componente del modelo y el propósito de cada una de las características utilizadas.

## Visión General: Un Ensemble Verdadero de Especialistas

El modelo está diseñado como un **ensemble real** de dos especialistas con roles complementarios y **no redundantes**:

1.  **La LSTM (Red Neuronal Recurrente Multi-Output):** Es el **"Analista de Regímenes de Mercado y Tendencias de Largo Plazo"**. Su tarea es identificar en qué régimen se encuentra el mercado (alcista, bajista, lateral, volátil) y predecir tendencias a múltiples horizontes temporales (5, 10, 20 días).
2.  **El XGBoost (Gradient Boosting):** Es el **"Decisor Táctico de Corto Plazo"**. Utiliza señales técnicas de corto plazo junto con el contexto de régimen proporcionado por la LSTM para tomar decisiones de timing precisas día a día.

---

## Componente 1: La LSTM - El Analista de Regímenes

### Rol en el Equipo
Identificar **regímenes de mercado** y **tendencias de medio/largo plazo** basándose en patrones temporales complejos. Responde a las preguntas: 
- "¿En qué régimen estamos? (bull/bear/sideways/volatile)"
- "¿Cuál es la tendencia esperada a 5, 10 y 20 días?"
- "¿Cuán confiado está el modelo en su clasificación de régimen?"

### Arquitectura Multi-Output (Por Separado)
-   **Entrada:** Ventanas deslizantes de **60 días consecutivos** (2-3 meses de trading). Cada día contiene 8 características enfocadas en **patrones de largo plazo**:
    - `returns`: Retornos logarítmicos diarios
    - `log_returns`: Retornos logarítmicos 
    - `volatility_5`: Volatilidad de corto plazo (5 días)
    - `volatility_20`: Volatilidad de largo plazo (20 días) - **Detección de cambios de régimen**
    - `volume_trend`: Ratio volumen/media móvil - **Divergencias precio-volumen**
    - `price_cycle`: Desviación del precio respecto a MA(50) - **Ciclos de mercado**
    - `seasonality_dow`: Día de la semana normalizado - **Estacionalidad**
    - `seasonality_month`: Mes del año normalizado - **Estacionalidad**

-   **Arquitectura (3 Capas LSTM + Multi-Head):**
    ```
    Input (60, 8) → LSTM(128) → Dropout(0.3) 
                 → LSTM(128) → Dropout(0.3)
                 → LSTM(64)  → Dropout(0.3)
                 → Shared Representation
                 
    Head 1 (Regime Classification):
        → Dense(64, relu) → Dense(4, softmax) → [P(bull), P(bear), P(sideways), P(volatile)]
    
    Head 2 (Trend 5d):
        → Dense(32, relu) → Dense(1, linear) → Predicted return at t+5
    
    Head 3 (Trend 10d):
        → Dense(32, relu) → Dense(1, linear) → Predicted return at t+10
    
    Head 4 (Trend 20d):
        → Dense(32, relu) → Dense(1, linear) → Predicted return at t+20
    ```

-   **Salida:** 8 valores por cada día:
    1. **4 probabilidades de régimen** (suman 1.0):
       - `lstm_regime_bull`: Probabilidad de mercado alcista
       - `lstm_regime_bear`: Probabilidad de mercado bajista
       - `lstm_regime_sideways`: Probabilidad de mercado lateral
       - `lstm_regime_volatile`: Probabilidad de mercado volátil
    2. **3 predicciones de tendencia**:
       - `lstm_trend_5d`: Retorno esperado a 5 días
       - `lstm_trend_10d`: Retorno esperado a 10 días
       - `lstm_trend_20d`: Retorno esperado a 20 días
    3. **1 métrica de confianza**:
       - `lstm_regime_confidence`: Máxima probabilidad de régimen (indica certeza)

### Clasificación de Regímenes
Los regímenes se determinan observando 20 días hacia adelante:

- **Bull (Alcista):** Retorno futuro > 3% y volatilidad normal
- **Bear (Bajista):** Retorno futuro < -3% y volatilidad normal
- **Sideways (Lateral):** Retorno futuro entre -3% y 3%
- **Volatile (Volátil):** Volatilidad > 1.5× mediana (independiente de dirección)

---

## Componente 2: XGBoost - El Decisor Táctico

### Rol en el Equipo
Tomar la decisión final de clasificación binaria (¿el precio subirá o bajará mañana?) integrando:
1. **Contexto de régimen** proporcionado por LSTM
2. **Señales técnicas de corto plazo** (momentum, volatilidad inmediata)
3. **Indicadores macroeconómicos** (VIX)

### Funcionamiento (Por Separado)
-   **Entrada:** Una "fotografía" del mercado en un día `t` con 17 features especializadas en **corto plazo**.
-   **Proceso (Gradient Boosting con Early Stopping):**
    - Construye árboles de decisión secuenciales
    - Cada árbol corrige errores de los anteriores
    - Early stopping (50 rondas) evita sobreajuste
    - Validación mensual tipo walk-forward (mínimo 24 meses de entrenamiento)
    - Filtra folds con <10 muestras o clase única
-   **Salida:** Probabilidad de retorno positivo en t+1

---

## Catálogo Completo de Features para XGBoost

### A. Indicadores Técnicos de Corto Plazo (7 features)
*Especializados en momentum y volatilidad inmediata - NO duplican LSTM*

-   `rsi`: **RSI(14).** Momentum de corto plazo, sobrecompra/sobreventa
-   `macd`, `macdh`, `macds`: **MACD(12,26,9).** Convergencia de medias, cambios de tendencia
-   `atr`: **ATR(14).** Volatilidad intradiaria (diferente de volatility_5/20 del LSTM)
-   `ema_short` / `ema_long`: **EMA(20) / EMA(50).** Cruces de medias móviles

### B. Indicador Macroeconómico (1 feature)
-   `close_vix_ratio`: **Precio / VIX.** Relación entre precio y miedo del mercado

### C. Señales LSTM de Régimen y Tendencia (8 features)
*El contexto estratégico que diferencia este ensemble*

**Probabilidades de Régimen (4):**
-   `lstm_regime_bull`: Probabilidad de estar en mercado alcista
-   `lstm_regime_bear`: Probabilidad de estar en mercado bajista
-   `lstm_regime_sideways`: Probabilidad de estar en mercado lateral
-   `lstm_regime_volatile`: Probabilidad de estar en mercado volátil

**Predicciones de Tendencia Multi-Horizonte (3):**
-   `lstm_trend_5d`: Tendencia esperada a 5 días (corto plazo)
-   `lstm_trend_10d`: Tendencia esperada a 10 días (medio plazo)
-   `lstm_trend_20d`: Tendencia esperada a 20 días (largo plazo)

**Confianza del Modelo (1):**
-   `lstm_regime_confidence`: Certeza en la clasificación de régimen (0-1)

---

## Funcionamiento en Conjunto: El Ensemble Real

El verdadero poder del sistema reside en la **especialización y complementariedad** de ambos modelos.

### Paso 1: Análisis de Regímenes (LSTM)
Para cada día del histórico, la **LSTM**:
1. Analiza los últimos 60 días de datos de largo plazo (ciclos, volatilidad, estacionalidad)
2. Genera 4 probabilidades de régimen (clasificación softmax)
3. Predice tendencias a 5, 10 y 20 días (regresión multi-horizonte)
4. Calcula confianza basada en probabilidad máxima

**Output:** 8 nuevas features que describen el **contexto estratégico del mercado**

### Paso 2: Recopilación de Inteligencia
Se crea una tabla donde cada día contiene:
- **Señales técnicas corto plazo** (RSI, MACD, ATR, EMAs) → Timing
- **Contexto LSTM** (régimen + tendencias multi-horizonte) → Estrategia
- **Macro** (VIX ratio) → Entorno de riesgo

### Paso 3: Decisión Táctica (XGBoost)
El **XGBoost** integra toda la información:

**Ejemplo 1 - Bull Trap Detection:**
- LSTM dice: `lstm_regime_bull=0.7`, `lstm_trend_5d=+2%`
- Pero XGBoost ve: `rsi=78` (sobrecompra), `close_vix_ratio` bajo (miedo alto)
- **Decisión:** Predice BAJADA (el régimen alcista está agotado a corto plazo)

**Ejemplo 2 - Sideways con Oportunidad:**
- LSTM dice: `lstm_regime_sideways=0.6`, `lstm_trend_20d=+1%`
- XGBoost ve: `macd` cruza al alza, `atr` bajo (baja volatilidad)
- **Decisión:** Predice SUBIDA (pequeño movimiento alcista dentro de lateral)

**Ejemplo 3 - Volatile Regime:**
- LSTM dice: `lstm_regime_volatile=0.8`, `lstm_regime_confidence=0.8`
- XGBoost ve señales técnicas contradictorias
- **Decisión:** Reduce confianza o evita trading (alta incertidumbre)

### Ventajas del Diseño

1. **No hay redundancia:** LSTM usa ventanas largas (60d) y features de largo plazo; XGBoost usa señales corto plazo
2. **Información complementaria:** Regímenes + timing = ensemble verdadero
3. **Especialización clara:**
   - LSTM → ¿Dónde estamos? ¿Hacia dónde vamos?
   - XGBoost → ¿Cuándo entrar/salir ahora mismo?
4. **Validación robusta:** Walk-forward mensual con filtrado de folds inválidos
5. **Visualización completa:** Gráficos de regímenes + accuracy por régimen

---

## Métricas de Evaluación

### LSTM (Multi-Task)
- **Régimen:** Accuracy de clasificación (4 clases)
- **Tendencias:** MSE para cada horizonte (5d, 10d, 20d)
- **Pesos de pérdida:** Régimen=1.0, cada tendencia=0.5

### XGBoost (Clasificación Binaria)
- **Métrica principal:** AUC-ROC
- **Métrica secundaria:** Accuracy
- **Validación:** Walk-forward mensual (últimos 12 meses, mínimo 24 meses de entrenamiento)
- **Early stopping:** 50 rondas sin mejora en AUC

### Feature Importance
Analiza importancia relativa de:
- Señales técnicas vs LSTM
- Qué probabilidad de régimen es más predictiva
- Qué horizonte de tendencia ayuda más
