# Arquitectura y Características del Modelo Híbrido

Este documento detalla el funcionamiento interno del pipeline de `ForecastTRADE`, explicando el rol de cada componente del modelo y el propósito de cada una de las características utilizadas.

## Visión General: Un Equipo de Especialistas

El modelo está diseñado como un equipo de dos especialistas con habilidades complementarias:

1.  **La LSTM (Red Neuronal Recurrente):** Es el **"Analista de Patrones Visuales"**. Su única tarea es mirar la secuencia histórica de precios (las "velas japonesas") y aprender los patrones visuales y la inercia del mercado.
2.  **El XGBoost (Gradient Boosting):** Es el **"Director de Decisiones Estratégicas"**. No mira los gráficos, sino que recibe un conjunto de datos y análisis (incluido el informe de la LSTM) para tomar una decisión final, sopesando todos los factores.

---

## Componente 1: La LSTM - El Analista de Patrones

### Rol en el Equipo
Detectar la **inercia y la tendencia** basándose únicamente en la secuencia de los datos brutos de precio y volumen. Responde a la pregunta: "Dado cómo se han movido los precios en los últimos 60 días, ¿hacia dónde es más probable que se dirija el precio mañana por puro impulso?"

### Funcionamiento (Por Separado)
-   **Entrada:** El modelo no ve un día a la vez. Su entrada son "ventanas" deslizantes de **60 días consecutivos**. Cada uno de esos 60 días contiene 5 datos: `Open`, `High`, `Low`, `Close` y `Volume`. Por tanto, su input es un tensor tridimensional con forma `[número_de_muestras, 60, 5]`.
-   **Proceso (Memoria y Secuencia):** Una LSTM es un tipo de red neuronal con una "memoria" interna. Procesa la secuencia de 60 días de forma ordenada, recordando qué pasó en los días anteriores para entender el contexto. Nuestra arquitectura es "apilada" (Stacked LSTM), lo que significa que tiene dos capas LSTM una encima de la otra, permitiéndole aprender patrones a diferentes escalas de tiempo (patrones de corto plazo en la primera capa, y patrones de más largo plazo en la segunda). Las capas de `Dropout` ayudan a evitar que el modelo "memorice" el ruido en lugar de aprender el patrón real.
-   **Salida:** Después de analizar una secuencia de 60 días, la LSTM genera un único número: su mejor estimación para el **precio de cierre del día siguiente (`t+1`)**.

### Características que Utiliza (Features)
La LSTM solo se alimenta de los datos más brutos del mercado:
1.  `open`: Precio de apertura.
2.  `high`: Precio más alto del día.
3.  `low`: Precio más bajo del día.
4.  `close`: Precio de cierre.
5.  `volume`: Volumen de transacciones.

---

## Componente 2: XGBoost - El Director de Decisiones

### Rol en el Equipo
Tomar la decisión final de clasificación binaria (¿el precio subirá o bajará mañana?) basándose en un análisis integral de toda la información disponible en un único instante de tiempo.

### Funcionamiento (Por Separado)
-   **Entrada:** A diferencia de la LSTM, XGBoost no ve secuencias. Ve una "fotografía" del mercado en un día `t`. Su entrada es una tabla donde cada fila es un día y cada columna es una de las características (features) que describimos a continuación.
-   **Proceso (Árboles de Decisión):** XGBoost (eXtreme Gradient Boosting) es un algoritmo que construye cientos o miles de "árboles de decisión" de forma secuencial. Un árbol de decisión es como un diagrama de flujo de preguntas "si/entonces" (ej. "SI el RSI > 70 Y el VIX < 20, ENTONCES predecir subida"). El primer árbol hace una predicción simple. El segundo árbol se entrena para corregir los errores del primero. El tercero corrige los errores de los dos anteriores, y así sucesivamente. Este método es extremadamente potente para encontrar relaciones complejas y no lineales en datos tabulares.
-   **Salida:** Una probabilidad entre 0 y 1, que indica la confianza del modelo en que el retorno del día siguiente sea positivo.

---

## Catálogo Completo de Features para XGBoost

XGBoost utiliza un conjunto de características mucho más rico y diverso.

### A. Indicadores Técnicos
*Calculados sobre los precios con la librería `pandas_ta`.*

-   `rsi`: **Índice de Fuerza Relativa.** Mide la velocidad y el cambio de los movimientos de precios. Un valor alto (>70) sugiere "sobrecompra" y uno bajo (<30) "sobreventa". Ayuda a medir el *momentum*.
-   `macd`, `macdh`, `macds`: **Convergencia/Divergencia de Medias Móviles.** Es un indicador de seguimiento de tendencia. Compara dos medias móviles exponenciales para señalar cambios en la dirección y fuerza de la tendencia.
-   `atr`: **Rango Verdadero Promedio.** Mide la volatilidad del mercado. Un ATR alto significa que el precio se mueve mucho durante el día. No indica dirección, solo magnitud del movimiento.
-   `ema_short` / `ema_long`: **Medias Móviles Exponenciales (de 20 y 50 días).** Suavizan los datos de precios para mostrar la tendencia subyacente. El cruce entre la media corta y la larga es una señal de trading clásica.

### B. Indicadores de Sentimiento (Simulado)
*Estos simulan el "ruido" informativo y el sentimiento del mercado.*

-   `sentiment`: Puntuación de sentimiento diaria (entre -1 y 1).
-   `sentiment_ma`: **Media Móvil de Sentimiento.** Suaviza la puntuación de sentimiento para capturar la tendencia general del "ánimo" del mercado en lugar del ruido diario.
-   `sentiment_lag_1/2/3`: **Sentimiento de días anteriores.** Permite al modelo comprobar si el sentimiento de ayer, anteayer, etc., tiene poder predictivo sobre el precio de mañana.
-   `sentiment_ma_lag_1/2/3`: **Media móvil de sentimiento de días anteriores.**

### C. Indicador Macroeconómico
-   `close_vix_ratio`: **Ratio Precio de Cierre / Índice VIX.** El VIX es el "índice del miedo" del mercado. Este ratio mide el precio del activo en relación con el miedo del mercado. Un ratio alto puede sugerir complacencia o apetito por el riesgo, mientras que uno bajo puede indicar aversión al riesgo.

### D. El Indicador Experto de la LSTM
-   `lstm_trend_prediction`: **La predicción de precio de la LSTM.** Esta es la característica clave de nuestro modelo híbrido. Es la opinión del "Analista de Patrones" sobre hacia dónde se dirige el precio basándose en la pura inercia gráfica.

---

## Funcionamiento en Conjunto: El Modelo Híbrido

El verdadero poder del sistema reside en cómo estos dos componentes colaboran.

1.  **Paso 1 (Análisis de Patrones):** Para cada día del histórico, la **LSTM** analiza la ventana de los 60 días anteriores y produce su predicción de precio para el día siguiente. Esta predicción se guarda como la nueva característica `lstm_trend_prediction`.

2.  **Paso 2 (Recopilación de Inteligencia):** Se crea una tabla de datos completa. Cada fila representa un día y contiene todos los indicadores: técnicos (RSI, MACD...), de sentimiento, macroeconómicos (Close/VIX) y, crucialmente, la `lstm_trend_prediction` generada en el paso anterior.

3.  **Paso 3 (Decisión Estratégica):** La tabla completa se entrega al **XGBoost**. El "Director de Decisiones" ahora tiene toda la información en su mesa.
    *   Puede que la LSTM diga "¡La tendencia visual es fuertemente alcista!" (`lstm_trend_prediction` es alto).
    *   Pero el XGBoost también verá que el VIX está por las nubes (`close_vix_ratio` es bajo) y que el RSI está en zona de sobrecompra.
    *   Considerando todos estos factores, el XGBoost puede decidir que la señal de la LSTM es una "trampa alcista" (bull trap) y optar por predecir una bajada, o viceversa.

Esta sinergia permite al modelo validar los patrones de tendencia visual de la LSTM contra un contexto de mercado más amplio, creando un sistema de pronóstico más robusto y menos propenso a ser engañado por el ruido.
