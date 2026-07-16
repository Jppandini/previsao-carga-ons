# ============================================
# Regressao de Carga com SVR (RBF) - Bateria com hiperparametros fixos
# Fonte: parquet da Curva de Carga do ONS | Treino deslizante 1/3/5 anos
# ============================================
#
# Roda as 18 janelas (6 periodos x 3 janelas de treino). Cada cenario usa os
# hiperparametros fixos definidos no trabalho (nao ha busca em grade). Para
# cada cenario:
#   - treino do SVR com os hiperparametros do cenario
#   - avaliacao no periodo de teste
#   - grafico real vs previsto (scatter) salvo em PASTA_GRAFICOS

# -------------- CONFIGURACAO --------------
# Bateria de testes: 6 periodos x 3 janelas de treino = 18 cenarios.
# Hiperparametros fixos por cenario (os utilizados no trabalho): C, gamma, epsilon
BATERIA_TESTES = [
    # --- 3 dias x 3 janelas ---
    # Sexta 06/06/2025
    {"nome": "Sexta 06/06/2025 (1 ano)",   "data_inicio_teste": "2025-06-06", "data_fim_teste": "2025-06-06", "inicio_treino": "2024-06-06", "C": 100, "gamma": 0.01,    "epsilon": 0.01},
    {"nome": "Sexta 06/06/2025 (3 anos)",  "data_inicio_teste": "2025-06-06", "data_fim_teste": "2025-06-06", "inicio_treino": "2022-06-06", "C": 10,  "gamma": 0.01,    "epsilon": 0.01},
    {"nome": "Sexta 06/06/2025 (5 anos)",  "data_inicio_teste": "2025-06-06", "data_fim_teste": "2025-06-06", "inicio_treino": "2020-06-06", "C": 10,  "gamma": "scale", "epsilon": 0.01},
    # Sabado 07/06/2025
    {"nome": "Sábado 07/06/2025 (1 ano)",  "data_inicio_teste": "2025-06-07", "data_fim_teste": "2025-06-07", "inicio_treino": "2024-06-07", "C": 100, "gamma": 0.01,    "epsilon": 0.01},
    {"nome": "Sábado 07/06/2025 (3 anos)", "data_inicio_teste": "2025-06-07", "data_fim_teste": "2025-06-07", "inicio_treino": "2022-06-07", "C": 10,  "gamma": "scale", "epsilon": 0.01},
    {"nome": "Sábado 07/06/2025 (5 anos)", "data_inicio_teste": "2025-06-07", "data_fim_teste": "2025-06-07", "inicio_treino": "2020-06-07", "C": 10,  "gamma": "scale", "epsilon": 0.01},
    # Domingo 08/06/2025
    {"nome": "Domingo 08/06/2025 (1 ano)", "data_inicio_teste": "2025-06-08", "data_fim_teste": "2025-06-08", "inicio_treino": "2024-06-08", "C": 10,  "gamma": "scale", "epsilon": 0.01},
    {"nome": "Domingo 08/06/2025 (3 anos)","data_inicio_teste": "2025-06-08", "data_fim_teste": "2025-06-08", "inicio_treino": "2022-06-08", "C": 10,  "gamma": "scale", "epsilon": 0.01},
    {"nome": "Domingo 08/06/2025 (5 anos)","data_inicio_teste": "2025-06-08", "data_fim_teste": "2025-06-08", "inicio_treino": "2020-06-08", "C": 10,  "gamma": "scale", "epsilon": 0.01},
    # --- 3 meses x 3 janelas ---
    # Junho/2025
    {"nome": "Junho/2025 (1 ano)",         "data_inicio_teste": "2025-06-01", "data_fim_teste": "2025-06-30", "inicio_treino": "2024-06-01", "C": 10,  "gamma": 0.01,    "epsilon": 0.01},
    {"nome": "Junho/2025 (3 anos)",        "data_inicio_teste": "2025-06-01", "data_fim_teste": "2025-06-30", "inicio_treino": "2022-06-01", "C": 10,  "gamma": 0.01,    "epsilon": 0.01},
    {"nome": "Junho/2025 (5 anos)",        "data_inicio_teste": "2025-06-01", "data_fim_teste": "2025-06-30", "inicio_treino": "2020-06-01", "C": 10,  "gamma": 0.01,    "epsilon": 0.01},
    # Setembro/2025
    {"nome": "Setembro/2025 (1 ano)",      "data_inicio_teste": "2025-09-01", "data_fim_teste": "2025-09-30", "inicio_treino": "2024-09-01", "C": 10,  "gamma": 0.01,    "epsilon": 0.01},
    {"nome": "Setembro/2025 (3 anos)",     "data_inicio_teste": "2025-09-01", "data_fim_teste": "2025-09-30", "inicio_treino": "2022-09-01", "C": 10,  "gamma": 0.01,    "epsilon": 0.01},
    {"nome": "Setembro/2025 (5 anos)",     "data_inicio_teste": "2025-09-01", "data_fim_teste": "2025-09-30", "inicio_treino": "2020-09-01", "C": 10,  "gamma": 0.01,    "epsilon": 0.01},
    # Dezembro/2025
    {"nome": "Dezembro/2025 (1 ano)",      "data_inicio_teste": "2025-12-01", "data_fim_teste": "2025-12-31", "inicio_treino": "2024-12-01", "C": 10,  "gamma": 0.01,    "epsilon": 0.01},
    {"nome": "Dezembro/2025 (3 anos)",     "data_inicio_teste": "2025-12-01", "data_fim_teste": "2025-12-31", "inicio_treino": "2022-12-01", "C": 10,  "gamma": 0.01,    "epsilon": 0.01},
    {"nome": "Dezembro/2025 (5 anos)",     "data_inicio_teste": "2025-12-01", "data_fim_teste": "2025-12-31", "inicio_treino": "2020-12-01", "C": 10,  "gamma": 0.01,    "epsilon": 0.01},
]


USE_LOG_TARGET = True
USE_EXTRA_LAGS = True
USE_YEARLY_SEASONALITY = True
USE_WEEKEND = True
USE_TIMESTAMP = True


# Pasta com os parquet do ONS (baixados manualmente)
PASTA_DADOS = 'dados'
# Pastas/arquivos de saida
PASTA_GRAFICOS = 'graficos_svr'
ARQUIVO_CSV = 'resumo_resultados_svr.csv'
# ------------------------------------------


# Imports
import os
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

from ons_data import carregar_curva_carga


# ----------------------------------
# Funcoes auxiliares
# ----------------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def calcular_periodo_treino(config):
    data_inicio_treino = pd.Timestamp(config['inicio_treino'])
    data_fim_treino = pd.Timestamp(config['data_inicio_teste']) - timedelta(days=1)
    return data_inicio_treino, data_fim_treino


# -----------------------------
# 1. Carga dos dados
# -----------------------------
print("="*60)
print("CARREGANDO CURVA DE CARGA DO ONS")
print("="*60)

df_bruto = carregar_curva_carga(PASTA_DADOS)


# -----------------------------
# 2. Pre-processamento
# -----------------------------
df_bruto = df_bruto.dropna(subset=['val_geracao'])
df_bruto['ano'] = df_bruto['ano'].astype(int)
df_bruto['mes'] = df_bruto['mes'].astype(int)
df_bruto['dia'] = df_bruto['dia'].astype(int)
df_bruto['hora'] = df_bruto['hora'].astype(int)

df_bruto['datetime'] = pd.to_datetime(
    df_bruto[['ano', 'mes', 'dia', 'hora']]
      .rename(columns={'ano': 'year', 'mes': 'month', 'dia': 'day', 'hora': 'hour'})
)

df_bruto['timestamp'] = df_bruto['datetime'].astype(np.int64) // 10**9
df_bruto['dayofweek'] = df_bruto['datetime'].dt.dayofweek
df_bruto['hour_sin'] = np.sin(2 * np.pi * df_bruto['hora'] / 24)
df_bruto['hour_cos'] = np.cos(2 * np.pi * df_bruto['hora'] / 24)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

df_bruto = df_bruto.sort_values('datetime').reset_index(drop=True)

df_bruto['lag1'] = df_bruto['val_geracao'].shift(1)
df_bruto['lag24'] = df_bruto['val_geracao'].shift(24)
df_bruto['roll_mean_24'] = df_bruto['val_geracao'].rolling(window=24, min_periods=24).mean()

if USE_EXTRA_LAGS:
    df_bruto['lag48'] = df_bruto['val_geracao'].shift(48)
    df_bruto['lag168'] = df_bruto['val_geracao'].shift(168)
    df_bruto['roll_mean_168'] = df_bruto['val_geracao'].rolling(window=168, min_periods=168).mean()

if USE_YEARLY_SEASONALITY:
    df_bruto['dayofyear'] = df_bruto['datetime'].dt.dayofyear
    df_bruto['doy_sin'] = np.sin(2 * np.pi * df_bruto['dayofyear'] / 365)
    df_bruto['doy_cos'] = np.cos(2 * np.pi * df_bruto['dayofyear'] / 365)

if USE_WEEKEND:
    df_bruto['is_weekend'] = df_bruto['dayofweek'].isin([5, 6]).astype(int)

lag_cols = ['lag1', 'lag24', 'roll_mean_24']
if USE_EXTRA_LAGS:
    lag_cols += ['lag48', 'lag168', 'roll_mean_168']

df_bruto = df_bruto.dropna(subset=lag_cols)
print(f"✅ Dataset preparado: {df_bruto.shape}")


# -----------------------------
# 3. Montagem da lista de features
# -----------------------------
feature_cols = []
if USE_TIMESTAMP:
    feature_cols.append('timestamp')
feature_cols += ['dayofweek', 'hour_sin', 'hour_cos']
if USE_YEARLY_SEASONALITY:
    feature_cols += ['doy_sin', 'doy_cos']
if USE_WEEKEND:
    feature_cols.append('is_weekend')
feature_cols += ['lag1', 'lag24', 'roll_mean_24']
if USE_EXTRA_LAGS:
    feature_cols += ['lag48', 'lag168', 'roll_mean_168']

target = "val_geracao"


# ============================================
# LOOP PRINCIPAL - BATERIA COM HIPERPARAMETROS FIXOS
# ============================================
os.makedirs(PASTA_GRAFICOS, exist_ok=True)

# Recomeca o CSV do zero a cada execucao: a bateria e reprodutivel e a
# numeracao dos testes e fixa (1..N), nao acumula entre rodadas.
if os.path.exists(ARQUIVO_CSV):
    os.remove(ARQUIVO_CSV)

print("\n" + "="*70)
print(f"BATERIA SVR COM HIPERPARÂMETROS FIXOS POR CENÁRIO")
print(f"Cenários a rodar: {len(BATERIA_TESTES)}")
print("="*70)

for idx_teste, config in enumerate(BATERIA_TESTES, 1):
    print("\n" + "█"*70)
    print(f"CENÁRIO {idx_teste}/{len(BATERIA_TESTES)}: {config['nome']}")
    print("█"*70)

    # ----- Periodos -----
    data_inicio_treino, data_fim_treino = calcular_periodo_treino(config)
    data_inicio_teste = pd.Timestamp(config['data_inicio_teste'])
    data_fim_teste = pd.Timestamp(config['data_fim_teste'] + " 23:59:59")
    data_fim_treino_full = pd.Timestamp(data_fim_treino.strftime('%Y-%m-%d') + " 23:59:59")

    print(f"📅 Treino: {data_inicio_treino.strftime('%d/%m/%Y')} a {data_fim_treino.strftime('%d/%m/%Y')}")
    print(f"📅 Teste:  {data_inicio_teste.strftime('%d/%m/%Y')} a {data_fim_teste.strftime('%d/%m/%Y')}")

    # ----- Split de dados -----
    train = df_bruto[(df_bruto['datetime'] >= data_inicio_treino) &
                     (df_bruto['datetime'] <= data_fim_treino_full)]
    test = df_bruto[(df_bruto['datetime'] >= data_inicio_teste) &
                    (df_bruto['datetime'] <= data_fim_teste)]

    if len(train) == 0 or len(test) == 0:
        print(f"⚠️  PULANDO: Conjunto de treino ou teste vazio")
        continue

    print(f"   Train original: {len(train)} | Test: {len(test)}")

    # ----- Scaling e alvo -----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test = scaler.transform(test[feature_cols])
    y_train = train[target].values
    y_test = test[target].values

    if USE_LOG_TARGET:
        y_train_tr = np.log1p(y_train)
    else:
        y_train_tr = y_train

    # ----- Hiperparametros fixos do cenario (definidos no trabalho) -----
    C_val = config['C']
    gamma_val = config['gamma']
    epsilon_val = config['epsilon']

    print(f"\n⚙️  Hiperparâmetros: C={C_val}, gamma={gamma_val}, epsilon={epsilon_val}")

    # ----- Treino do SVR -----
    tempo_cenario_inicio = time.time()

    model = SVR(kernel='rbf', C=C_val, gamma=gamma_val, epsilon=epsilon_val)
    model.fit(X_train, y_train_tr)

    y_pred_tr = model.predict(X_test)
    if USE_LOG_TARGET:
        y_pred = np.expm1(y_pred_tr)
    else:
        y_pred = y_pred_tr

    tempo_cenario_total = time.time() - tempo_cenario_inicio

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"\n📊 Resultados: R²={r2:.4f} | RMSE={rmse:.2f} | MAPE={mape:.2f}% | Tempo={tempo_cenario_total:.1f}s")

    # ----- Numero do teste fixo (= indice do cenario) para reprodutibilidade -----
    numero_teste = idx_teste

    resumo = {
        'Teste': f'Teste {numero_teste}',
        'Tipo Teste': 'SVR (RBF)',
        'Período Treino': f"{train['datetime'].min().strftime('%d/%m/%Y')} a {train['datetime'].max().strftime('%d/%m/%Y')}",
        'Período Teste': f"{test['datetime'].min().strftime('%d/%m/%Y')} a {test['datetime'].max().strftime('%d/%m/%Y')}",
        'C': C_val,
        'Gamma': gamma_val,
        'Epsilon': epsilon_val,
        'R²': round(r2, 4),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2),
        'Tempo de Processamento (s)': round(tempo_cenario_total, 2)
    }

    arquivo_existe = os.path.exists(ARQUIVO_CSV)
    pd.DataFrame([resumo]).to_csv(
        ARQUIVO_CSV, mode='a', header=not arquivo_existe,
        index=False, sep=';', decimal=','
    )
    print(f"✅ Resumo salvo no CSV (Teste {numero_teste})")

    # ----- Gera e salva grafico -----
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, c='blue', alpha=0.7, s=50,
                edgecolors='black', linewidth=0.5)

    min_val = float(min(y_test.min(), y_pred.min()))
    max_val = float(max(y_test.max(), y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], 'b-', linewidth=3, label='Previsão Perfeita')
    plt.fill_between([min_val, max_val],
                     [min_val * 0.9, max_val * 0.9], [min_val * 1.1, max_val * 1.1],
                     alpha=0.1, color='green', label='±10%')
    plt.fill_between([min_val, max_val],
                     [min_val * 0.8, max_val * 0.8], [min_val * 1.2, max_val * 1.2],
                     alpha=0.05, color='orange', label='±20%')

    plt.xlabel("Valores Reais (MWmed)", fontsize=12, fontweight='bold')
    plt.ylabel("Valores Previstos (MWmed)", fontsize=12, fontweight='bold')
    plt.title(f"Teste {numero_teste} - SVR\nR² = {r2:.3f} | RMSE = {rmse:.1f} MWmed | MAPE = {mape:.2f}%",
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    caminho_grafico = os.path.join(PASTA_GRAFICOS, f'teste_{numero_teste:03d}_svr.png')
    plt.savefig(caminho_grafico, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Gráfico salvo: {caminho_grafico}")


print("\n" + "="*70)
print(f"✅ BATERIA CONCLUÍDA: {len(BATERIA_TESTES)} cenários processados")
print(f"📄 Resumo principal: {ARQUIVO_CSV}")
print("="*70)
