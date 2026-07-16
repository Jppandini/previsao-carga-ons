# ============================================
# Rede Neural Artificial (RNA) - Bateria de Testes com hiperparametros fixos
# Fonte: parquet da Curva de Carga do ONS | Treino deslizante 1/3/5 anos
# ============================================
#
# Roda as 18 janelas (6 periodos x 3 janelas de treino). Cada cenario usa os
# hiperparametros fixos definidos no trabalho (nao ha grid search). Para cada
# cenario:
#   - treino com os hiperparametros do cenario e avaliacao no periodo de teste
#   - grafico real vs previsto (scatter) salvo em PASTA_GRAFICOS
# Nos cenarios de dezembro tambem salva a curva de aprendizado (perda e R2 por
# epoca) e o historico .npz em PASTA_HISTORICOS.

# -------------- CONFIGURACAO --------------
# Bateria de testes: 6 periodos x 3 janelas de treino = 18 cenarios.
# Hiperparametros fixos por cenario (os utilizados no trabalho):
#   dropout, learning_rate, batch_size, huber_delta
BATERIA_TESTES = [
    # --- 3 dias x 3 janelas ---
    # Sexta 06/06/2025
    {"nome": "Sexta 06/06/2025 (1 ano)",  "data_inicio_teste": "2025-06-06", "data_fim_teste": "2025-06-06", "inicio_treino": "2024-06-06", "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    {"nome": "Sexta 06/06/2025 (3 anos)", "data_inicio_teste": "2025-06-06", "data_fim_teste": "2025-06-06", "inicio_treino": "2022-06-06", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    {"nome": "Sexta 06/06/2025 (5 anos)", "data_inicio_teste": "2025-06-06", "data_fim_teste": "2025-06-06", "inicio_treino": "2020-06-06", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    # Sabado 07/06/2025
    {"nome": "Sábado 07/06/2025 (1 ano)",  "data_inicio_teste": "2025-06-07", "data_fim_teste": "2025-06-07", "inicio_treino": "2024-06-07", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    {"nome": "Sábado 07/06/2025 (3 anos)", "data_inicio_teste": "2025-06-07", "data_fim_teste": "2025-06-07", "inicio_treino": "2022-06-07", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    {"nome": "Sábado 07/06/2025 (5 anos)", "data_inicio_teste": "2025-06-07", "data_fim_teste": "2025-06-07", "inicio_treino": "2020-06-07", "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    # Domingo 08/06/2025
    {"nome": "Domingo 08/06/2025 (1 ano)",  "data_inicio_teste": "2025-06-08", "data_fim_teste": "2025-06-08", "inicio_treino": "2024-06-08", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    {"nome": "Domingo 08/06/2025 (3 anos)", "data_inicio_teste": "2025-06-08", "data_fim_teste": "2025-06-08", "inicio_treino": "2022-06-08", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    {"nome": "Domingo 08/06/2025 (5 anos)", "data_inicio_teste": "2025-06-08", "data_fim_teste": "2025-06-08", "inicio_treino": "2020-06-08", "dropout": 0.2, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    # --- 3 meses x 3 janelas ---
    # Junho/2025
    {"nome": "Junho/2025 (1 ano)",  "data_inicio_teste": "2025-06-01", "data_fim_teste": "2025-06-30", "inicio_treino": "2024-06-01", "dropout": 0.1, "learning_rate": 0.010, "batch_size": 32,  "huber_delta": 0.5},
    {"nome": "Junho/2025 (3 anos)", "data_inicio_teste": "2025-06-01", "data_fim_teste": "2025-06-30", "inicio_treino": "2022-06-01", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    {"nome": "Junho/2025 (5 anos)", "data_inicio_teste": "2025-06-01", "data_fim_teste": "2025-06-30", "inicio_treino": "2020-06-01", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    # Setembro/2025
    {"nome": "Setembro/2025 (1 ano)",  "data_inicio_teste": "2025-09-01", "data_fim_teste": "2025-09-30", "inicio_treino": "2024-09-01", "dropout": 0.1, "learning_rate": 0.010, "batch_size": 32,  "huber_delta": 1.0},
    {"nome": "Setembro/2025 (3 anos)", "data_inicio_teste": "2025-09-01", "data_fim_teste": "2025-09-30", "inicio_treino": "2022-09-01", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    {"nome": "Setembro/2025 (5 anos)", "data_inicio_teste": "2025-09-01", "data_fim_teste": "2025-09-30", "inicio_treino": "2020-09-01", "dropout": 0.2, "learning_rate": 0.001, "batch_size": 128, "huber_delta": 1.0},
    # Dezembro/2025
    {"nome": "Dezembro/2025 (1 ano)",  "data_inicio_teste": "2025-12-01", "data_fim_teste": "2025-12-31", "inicio_treino": "2024-12-01", "dropout": 0.1, "learning_rate": 0.010, "batch_size": 32,  "huber_delta": 1.0},
    {"nome": "Dezembro/2025 (3 anos)", "data_inicio_teste": "2025-12-01", "data_fim_teste": "2025-12-31", "inicio_treino": "2022-12-01", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
    {"nome": "Dezembro/2025 (5 anos)", "data_inicio_teste": "2025-12-01", "data_fim_teste": "2025-12-31", "inicio_treino": "2020-12-01", "dropout": 0.3, "learning_rate": 0.001, "batch_size": 64,  "huber_delta": 1.0},
]

# Proporcao da validacao dentro do treino (para early stopping)
VAL_SPLIT_RATIO = 0.15

# Epocas (fixas)
EPOCHS_FINAL = 150

# Pasta com os parquet do ONS (baixados manualmente)
PASTA_DADOS = 'dados'
# Pastas/arquivos de saida
PASTA_GRAFICOS = 'graficos_rna'
ARQUIVO_CSV = 'resumo_resultados_rna.csv'
PASTA_HISTORICOS = 'historico'
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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ons_data import carregar_curva_carga

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.losses import Huber
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


# ----------------------------------
# Funcoes auxiliares
# ----------------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def construir_modelo(input_dim, dropout_rate, learning_rate, huber_delta):
    """Constroi o modelo RNA com os hiperparametros especificados."""
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate * 0.67),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        loss=Huber(delta=huber_delta),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['mae']
    )
    return model


def calcular_periodo_treino(config):
    """Retorna inicio e fim do treino a partir da config do teste."""
    data_inicio_treino = pd.Timestamp(config['inicio_treino'])
    data_fim_treino = pd.Timestamp(config['data_inicio_teste']) - timedelta(days=1)
    return data_inicio_treino, data_fim_treino


class R2PorEpoca(Callback):
    """Registra R2 de treino e validacao ao fim de cada epoca (escala log).

    keras.metrics.R2Score nao registra em algumas versoes do Keras; este
    callback calcula o R2 por epoca com sklearn, independente da versao.
    """
    def __init__(self, X_tr, y_tr, X_vl, y_vl):
        super().__init__()
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_vl, self.y_vl = X_vl, y_vl
        self.r2_treino = []
        self.r2_val = []

    def on_epoch_end(self, epoch, logs=None):
        p_tr = self.model.predict(self.X_tr, verbose=0).flatten()
        p_vl = self.model.predict(self.X_vl, verbose=0).flatten()
        self.r2_treino.append(r2_score(self.y_tr, p_tr))
        self.r2_val.append(r2_score(self.y_vl, p_vl))


# -----------------------------
# 1. Carga dos dados (uma unica vez para toda a bateria)
# -----------------------------
print("="*60)
print("CARREGANDO CURVA DE CARGA DO ONS")
print("="*60)

df_bruto = carregar_curva_carga(PASTA_DADOS)


# -----------------------------
# 2. Pre-processamento (uma unica vez)
# -----------------------------
df_bruto = df_bruto.dropna(subset=['val_geracao'])
df_bruto['ano'] = df_bruto['ano'].astype(int)
df_bruto['mes'] = df_bruto['mes'].astype(int)
df_bruto['dia'] = df_bruto['dia'].astype(int)
df_bruto['hora'] = df_bruto['hora'].astype(int)

df_bruto['datetime'] = pd.to_datetime(
    df_bruto[['ano', 'mes', 'dia', 'hora']].rename(
        columns={'ano': 'year', 'mes': 'month', 'dia': 'day', 'hora': 'hour'}
    )
)
df_bruto = df_bruto.sort_values('datetime').reset_index(drop=True)

# Features temporais
df_bruto['timestamp'] = df_bruto['datetime'].astype(np.int64) // 10**9
df_bruto['dayofweek'] = df_bruto['datetime'].dt.dayofweek
df_bruto['hour_sin'] = np.sin(2 * np.pi * df_bruto['hora'] / 24)
df_bruto['hour_cos'] = np.cos(2 * np.pi * df_bruto['hora'] / 24)

# Lags e medias moveis
df_bruto['lag1'] = df_bruto['val_geracao'].shift(1)
df_bruto['lag24'] = df_bruto['val_geracao'].shift(24)
df_bruto['roll_mean_24'] = df_bruto['val_geracao'].rolling(window=24, min_periods=24).mean()
df_bruto = df_bruto.dropna(subset=['lag1', 'lag24', 'roll_mean_24'])

print(f"✅ Dataset preparado: {df_bruto.shape}")


# -----------------------------
# 3. Features e alvo
# -----------------------------
feature_cols = ['timestamp', 'dayofweek', 'hour_sin', 'hour_cos',
                'lag1', 'lag24', 'roll_mean_24']
target = "val_geracao"


# ============================================
# LOOP PRINCIPAL - BATERIA DE TESTES
# ============================================
os.makedirs(PASTA_GRAFICOS, exist_ok=True)
os.makedirs(PASTA_HISTORICOS, exist_ok=True)

# Recomeca o CSV do zero a cada execucao: a bateria e reprodutivel e a
# numeracao dos testes e fixa (1..N), nao acumula entre rodadas.
if os.path.exists(ARQUIVO_CSV):
    os.remove(ARQUIVO_CSV)

print("\n" + "="*70)
print(f"INICIANDO BATERIA DE {len(BATERIA_TESTES)} TESTES")
print(f"Hiperparâmetros fixos por cenário (sem grid search)")
print("="*70)

for idx_teste, config in enumerate(BATERIA_TESTES, 1):
    print("\n" + "█"*70)
    print(f"TESTE {idx_teste}/{len(BATERIA_TESTES)}: {config['nome']}")
    print("█"*70)

    nome_cenario = config['nome']
    eh_dezembro = 'dezembro' in nome_cenario.lower()

    # ----- Calcular periodo de treino deslizante -----
    data_inicio_treino, data_fim_treino = calcular_periodo_treino(config)
    data_inicio_teste = pd.Timestamp(config['data_inicio_teste'])
    data_fim_teste = pd.Timestamp(config['data_fim_teste'] + " 23:59:59")
    data_fim_treino_full = pd.Timestamp(data_fim_treino.strftime('%Y-%m-%d') + " 23:59:59")

    print(f"📅 Treino: {data_inicio_treino.strftime('%d/%m/%Y')} a {data_fim_treino.strftime('%d/%m/%Y')}")
    print(f"📅 Teste:  {data_inicio_teste.strftime('%d/%m/%Y')} a {data_fim_teste.strftime('%d/%m/%Y')}")

    # ----- Split de dados -----
    train_full = df_bruto[(df_bruto['datetime'] >= data_inicio_treino) &
                          (df_bruto['datetime'] <= data_fim_treino_full)].copy()
    test = df_bruto[(df_bruto['datetime'] >= data_inicio_teste) &
                    (df_bruto['datetime'] <= data_fim_teste)].copy()

    if len(train_full) == 0 or len(test) == 0:
        print(f"⚠️  PULANDO: Conjunto de treino ou teste vazio para {config['nome']}")
        continue

    n_total = len(train_full)
    n_val = int(n_total * VAL_SPLIT_RATIO)
    train = train_full.iloc[:-n_val].copy()
    validation = train_full.iloc[-n_val:].copy()

    print(f"   Train: {len(train)} | Val: {len(validation)} | Test: {len(test)}")

    # ----- Preparar arrays -----
    X_train = train[feature_cols].values
    X_val = validation[feature_cols].values
    X_test = test[feature_cols].values
    y_train = train[target].values
    y_val = validation[target].values
    y_test = test[target].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    # ----- Hiperparametros fixos do cenario (definidos no trabalho) -----
    params = {
        'dropout_rate': config['dropout'],
        'learning_rate': config['learning_rate'],
        'batch_size': config['batch_size'],
        'huber_delta': config['huber_delta'],
    }
    print(f"\n⚙️  Hiperparâmetros: DR={params['dropout_rate']}, LR={params['learning_rate']}, "
          f"BS={params['batch_size']}, HD={params['huber_delta']}")

    # ----- Treino -----
    # Nos cenarios de dezembro anexamos o callback de R2 por epoca para a curva
    # de aprendizado; nos demais nao, para nao custar tempo.
    tempo_inicio = time.time()

    print(f"🚀 Treinando modelo com {EPOCHS_FINAL} épocas...")
    model = construir_modelo(
        X_train.shape[1],
        params['dropout_rate'],
        params['learning_rate'],
        params['huber_delta']
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=0)
    ]

    r2cb = None
    if eh_dezembro:
        r2cb = R2PorEpoca(X_train, y_train_log, X_val, y_val_log)
        callbacks.append(r2cb)

    history = model.fit(
        X_train, y_train_log,
        validation_data=(X_val, y_val_log),
        epochs=EPOCHS_FINAL,
        batch_size=params['batch_size'],
        callbacks=callbacks, verbose=0
    )

    # ----- Predicao e Avaliacao -----
    y_pred_log = model.predict(X_test, verbose=0).flatten()
    y_pred_nn = np.expm1(y_pred_log)

    tempo_fim = time.time()
    tempo_processamento = tempo_fim - tempo_inicio

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    r2 = r2_score(y_test, y_pred_nn)
    mape = mean_absolute_percentage_error(y_test, y_pred_nn)

    print(f"\n📊 Resultados: R²={r2:.4f} | RMSE={rmse:.2f} | MAPE={mape:.2f}% | Tempo={tempo_processamento:.1f}s")

    # ----- Numero do teste fixo (= indice do cenario) para reprodutibilidade -----
    numero_teste = idx_teste

    epocas_treinadas = len(history.history['loss'])

    resumo = {
        'Teste': f'Teste {numero_teste}',
        'Tipo Teste': 'RNA',
        'Período Treino': f"{train_full['datetime'].min().strftime('%d/%m/%Y')} a {train_full['datetime'].max().strftime('%d/%m/%Y')}",
        'Período Teste': f"{test['datetime'].min().strftime('%d/%m/%Y')} a {test['datetime'].max().strftime('%d/%m/%Y')}",
        'Dropout': params['dropout_rate'],
        'LR': params['learning_rate'],
        'Batch Size': params['batch_size'],
        'Delta Huber': params['huber_delta'],
        'Épocas': epocas_treinadas,
        'R²': round(r2, 4),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2),
        'Tempo de Processamento (s)': round(tempo_processamento, 2)
    }

    arquivo_existe = os.path.exists(ARQUIVO_CSV)
    pd.DataFrame([resumo]).to_csv(
        ARQUIVO_CSV, mode='a', header=not arquivo_existe,
        index=False, sep=';', decimal=','
    )
    print(f"✅ Resumo salvo no CSV (Teste {numero_teste})")

    # ----- Gerar e salvar grafico real vs previsto -----
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred_nn, c='blue', alpha=0.7, s=50,
                edgecolors='black', linewidth=0.5)

    min_val = float(min(y_test.min(), y_pred_nn.min()))
    max_val = float(max(y_test.max(), y_pred_nn.max()))
    plt.plot([min_val, max_val], [min_val, max_val], 'b-', linewidth=3, label='Previsão Perfeita')
    plt.fill_between([min_val, max_val],
                     [min_val * 0.9, max_val * 0.9], [min_val * 1.1, max_val * 1.1],
                     alpha=0.1, color='green', label='±10%')
    plt.fill_between([min_val, max_val],
                     [min_val * 0.8, max_val * 0.8], [min_val * 1.2, max_val * 1.2],
                     alpha=0.05, color='orange', label='±20%')

    plt.xlabel("Valores Reais (MWmed)", fontsize=12, fontweight='bold')
    plt.ylabel("Valores Previstos (MWmed)", fontsize=12, fontweight='bold')
    plt.title(f"Teste {numero_teste} - RNA\nR² = {r2:.3f} | RMSE = {rmse:.1f} MWmed | MAPE = {mape:.2f}%",
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

    caminho_grafico = os.path.join(PASTA_GRAFICOS, f'teste_{numero_teste:03d}_rna.png')
    plt.savefig(caminho_grafico, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Gráfico salvo: {caminho_grafico}")

    # ----- Curva de aprendizado (somente dezembro) -----
    if eh_dezembro:
        loss_treino = np.array(history.history['loss'])
        loss_val = np.array(history.history['val_loss'])
        r2_treino = np.array(r2cb.r2_treino)   # vem do callback, nao do history
        r2_val = np.array(r2cb.r2_val)
        melhor_epoca = int(np.argmin(loss_val)) + 1

        caminho_npz = os.path.join(PASTA_HISTORICOS, f'historico_teste_{numero_teste:03d}.npz')
        np.savez(
            caminho_npz,
            loss_treino=loss_treino, loss_val=loss_val,
            r2_treino=r2_treino, r2_val=r2_val,
            melhor_epoca=melhor_epoca, nome_cenario=nome_cenario
        )
        print(f"💾 [Dezembro] Histórico salvo: {caminho_npz}")

        epocas = np.arange(1, len(loss_treino) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(epocas, loss_treino, color='#1f4e79', linewidth=1.5, label='Treino')
        ax1.plot(epocas, loss_val, color='#c0504d', linewidth=1.5, label='Validação')
        ax1.axvline(melhor_epoca, color='gray', linestyle='--', linewidth=1.2,
                    label=f'Melhor época ({melhor_epoca})')
        ax1.set_xlabel('Época', fontsize=10)
        ax1.set_ylabel('Perda (Huber)', fontsize=10)
        ax1.set_title('Evolução da Perda', fontsize=11, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(fontsize=8.5)

        ax2.plot(epocas, r2_treino, color='#1f4e79', linewidth=1.5, label='Treino')
        ax2.plot(epocas, r2_val, color='#c0504d', linewidth=1.5, label='Validação')
        ax2.axvline(melhor_epoca, color='gray', linestyle='--', linewidth=1.2,
                    label=f'Melhor época ({melhor_epoca})')
        ax2.set_xlabel('Época', fontsize=10)
        ax2.set_ylabel('R²', fontsize=10)
        ax2.set_title('Evolução do R²', fontsize=11, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.set_ylim(0, 1.02)   # R2 costuma ficar perto de 1; recorta para leitura
        ax2.legend(fontsize=8.5, loc='lower right')

        fig.suptitle(f"Teste {numero_teste} - RNA - {nome_cenario}",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        caminho_curva = os.path.join(PASTA_HISTORICOS, f'curva_aprendizado_teste_{numero_teste:03d}.png')
        plt.savefig(caminho_curva, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 [Dezembro] Curva salva: {caminho_curva}")


print("\n" + "="*70)
print(f"✅ BATERIA DE TESTES CONCLUÍDA: {len(BATERIA_TESTES)} testes processados")
print("="*70)
