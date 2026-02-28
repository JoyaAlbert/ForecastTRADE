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
- `--seq-encoder`: `off` | `lstm` | `tcn`.
- `--target-coverage`: ratio objetivo de cobertura de señales (ej. `0.12`).
- `--threshold-search`: `grid` | `quantile`.
- `--trade-ratio-floor`: cobertura mínima de trades por fold.
- `--conservative-th-min`: piso mínimo conservador para `threshold_buy`.
- `--conservative-th-max`: techo máximo conservador para `threshold_buy`.
- `--xgb-stochastic-mode`: `on` | `off`.
- `--xgb-candidate-budget`: `low` | `medium` | `high`.
- `--encoder-ablation`: `on` | `off`.
- `--benchmark-tickers`: lista separada por comas (ej. `MSFT,AAPL,NVDA`).
- `--robust-std-weight`: penalización por dispersión en ranking robusto.
- `--net-return-floor-fold`: piso de retorno neto por fold en tuning.
- `--no-ui`: desactiva selección interactiva.

Ver ayuda:

```bash
python run.py --help
```

Benchmark multi-seed + multi-config (selección real de defaults robustos):

```bash
python scripts/benchmark_defaults.py \
  --tickers MSFT,AAPL,NVDA \
  --seeds 42,77,101,123,777 \
  --plots none --profile off \
  --encoder-ablation on \
  --xgb-stochastic-mode on \
  --xgb-candidate-budget medium
```

Artefactos generados:
- `out/benchmark_defaults_grid.json` (detalle por corrida y ranking)
- `out/benchmark_defaults_grid.csv` (resumen tabular por candidato)
- `out/recommended_default_config.yaml` (config recomendada)

## Configuración externa (`configs/default.yaml`)

El pipeline admite configuración versionada por YAML. Puedes pasarla con `--config` o usar la por defecto en `configs/default.yaml`.

Claves relevantes:

- `SEED`
- `OBJECTIVE`
- `RISK_PROFILE`
- `CV_SCHEME`
- `SEQ_ENCODER`
- `THRESHOLD_SEARCH_MODE`
- `TARGET_COVERAGE_RATIO`
- `NET_SHARPE_TURNOVER_PENALTY`
- `TRADE_RATIO_FLOOR`
- `CONSERVATIVE_MIN_BUY_THRESHOLD_MIN`
- `CONSERVATIVE_MIN_BUY_THRESHOLD_MAX`
- `MAX_CALIBRATION_ECE_ALERT`
- `COST_BPS`
- `SLIPPAGE_BPS`
- `MIN_COVERAGE_RATIO`
- `ENABLE_STABILITY_TUNING`
- `ENABLE_PROBA_CALIBRATION`
- `CALIBRATION_METHOD`
- `CALIBRATION_MIN_SAMPLES`
- `USE_SYNTHETIC_SAMPLING`
- `XGB_PARAMS` (bloque completo de hiperparámetros)
- `BENCHMARK_TICKERS`
- `NET_RETURN_FLOOR_PER_FOLD`
- `MIN_VALID_FOLDS_RATIO`
- `ROBUST_SCORE_STD_WEIGHT`
- `ENCODER_ABLATION_ENABLED`
- `ENCODER_REQUIRED_DELTA_ROBUST_SCORE`
- `ENCODER_REQUIRED_DELTA_NET_SHARPE`

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
