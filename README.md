# ForecastTRADE v2.1

Sistema híbrido **LSTM + XGBoost** para predicción táctica de operaciones con validación temporal, calibración de probabilidades y recomendaciones operativas basadas en utilidad esperada.

## Características principales

- Arquitectura híbrida: LSTM como extractor de señales temporales + XGBoost como clasificador final.
- Esquemas de validación temporal:
  - `sliding` (ventana deslizante).
  - `purged` (purged + embargo para reducir leakage temporal).
- Optimización orientada a trading:
  - Objetivos CLI: `sharpe_net`, `return`, `max_winrate`.
  - Costes y slippage incluidos en métricas netas.
- Calibración de probabilidad por fold con métricas:
  - `calibration_error` (ECE).
  - `brier`.
- Motor de recomendación conservador por utilidad esperada (EV).
- Logging acumulativo enriquecido en `out/runs_log.json`.

## Requisitos

- Python 3.10+
- Entorno virtual recomendado
- Dependencias en `requirements.txt`

Instalación rápida:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución rápida

```bash
# Modo rápido sin plots
python run.py --mode fast --plots none --ticker MSFT --no-ui

# Modo completo con configuración por defecto
python run.py --mode full --plots final --ticker MSFT --no-ui
```

## CLI actual

```bash
python run.py [opciones]
```

Opciones principales:

- `--ticker`: símbolo (ej: `MSFT`, `AAPL`, `NVDA`).
- `--mode`: `full` | `fast`.
- `--plots`: `none` | `final` | `all`.
- `--cache`: `on` | `off`.
- `--profile`: `on` | `off`.
- `--config`: ruta a YAML de configuración.
- `--objective`: `sharpe_net` | `return` | `max_winrate`.
- `--risk-profile`: `conservative` | `balanced` | `aggressive`.
- `--cost-bps`: coste por transacción en bps.
- `--slippage-bps`: slippage en bps.
- `--cv-scheme`: `sliding` | `purged`.
- `--no-ui`: desactiva selección interactiva.

Ver ayuda:

```bash
python run.py --help
```

## Configuración externa (`configs/default.yaml`)

El pipeline admite configuración versionada por YAML. Puedes pasarla con `--config` o usar la por defecto en `configs/default.yaml`.

Claves relevantes:

- `SEED`
- `OBJECTIVE`
- `RISK_PROFILE`
- `CV_SCHEME`
- `COST_BPS`
- `SLIPPAGE_BPS`
- `MIN_COVERAGE_RATIO`
- `ENABLE_STABILITY_TUNING`
- `ENABLE_PROBA_CALIBRATION`
- `CALIBRATION_METHOD`
- `CALIBRATION_MIN_SAMPLES`
- `USE_SYNTHETIC_SAMPLING`
- `XGB_PARAMS` (bloque completo de hiperparámetros)

Ejemplo de uso:

```bash
python run.py --config configs/default.yaml --ticker MSFT --no-ui
```

## Ejemplos recomendados

```bash
# Perfil conservador con optimización de Sharpe neto y CV purged
python run.py \
  --mode full \
  --plots final \
  --ticker MSFT \
  --objective sharpe_net \
  --risk-profile conservative \
  --cost-bps 20 \
  --slippage-bps 5 \
  --cv-scheme purged \
  --no-ui

# Perfil balanceado priorizando retorno
python run.py \
  --mode full \
  --plots final \
  --ticker NVDA \
  --objective return \
  --risk-profile balanced \
  --cv-scheme sliding \
  --no-ui
```

## Salidas y artefactos

Directorio `out/`:

- `runs_log.json`: historial acumulado de corridas.
- `runs_summary.csv`: resumen tabular exportable.
- Visualizaciones `lstm_xgboost_hybrid_run_*` y `future_forecast_run_*` (según `--plots`).

Métricas clave registradas:

- `auc`, `auc_pr`, `accuracy`, `f1`
- `sharpe_ratio`, `max_drawdown`
- `net_sharpe`, `net_return`, `turnover`
- `coverage_ratio`
- `calibration_error`, `brier`
- `recommendation_quality`

## Recomendación operativa (estado)

La recomendación final usa probabilidad agregada + EV neta y emite estados como:

- `NO_TRADE`
- `WATCHLIST`
- `ENTER_SMALL`
- `ENTER_FULL`

## Notas de validación

Tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
```

Incluye pruebas para:

- split temporal purged sin solape
- calibración (ECE/Brier)
- optimización de umbrales
- motor de recomendación

## Documentación técnica

Para contexto de arquitectura original y componentes base:

- `MODEL_ARCHITECTURE.md`
