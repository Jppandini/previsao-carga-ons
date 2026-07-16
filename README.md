# Previsão da Curva de Carga do SIN — RNA e SVR

Previsão da carga horária do Sistema Interligado Nacional (SIN) usando dois
modelos de aprendizado de máquina, treinados sobre janelas temporais
deslizantes (1, 3 e 5 anos):

- **RNA** — Rede Neural Artificial (Keras/TensorFlow) com *grid search* de
  hiperparâmetros e *early stopping*.
- **SVR** — Support Vector Regression com kernel RBF e busca em grade.

Cada modelo é avaliado em **18 cenários** (6 períodos de teste × 3 janelas de
treino), usando **hiperparâmetros fixos por cenário** (os definidos no
trabalho — não há busca em grade). Cada cenário gera um gráfico *real vs.
previsto* e uma linha no CSV com o resumo das métricas (R², RMSE, MAPE, tempo
de processamento).

## Fonte dos dados

Curva de Carga Horária do ONS — dados abertos:
<https://dados.ons.org.br/dataset/curva-carga>

Os arquivos são parquet anuais no padrão:

```
https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/curva-carga-ho/CURVA_CARGA_<ANO>.parquet
```

Cada arquivo traz a carga horária **por subsistema** (N, NE, S, SE/CO). O
loader ([ons_data.py](ons_data.py)) soma os subsistemas em cada instante
horário para obter a carga total do SIN.

## Estrutura

```
previsao-carga-ons/
├── ons_data.py        # loader: lê ./dados/*.parquet → carga total horária
├── treino_rna.py      # treino RNA nas 18 janelas + gráficos
├── treino_svr.py      # treino SVR nas 18 janelas + gráficos
├── requirements.txt
└── dados/             # coloque aqui os parquet do ONS (não versionados)
```

## Como usar

### 1. Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** no `requirements.txt`, troque
> `tensorflow==2.13.0` por `tensorflow-macos==2.13.0`.

### 2. Baixar os dados

Baixe os parquet dos anos necessários para a pasta `dados/`. Para rodar todas
as janelas (incluindo as de 5 anos, que começam em 2020) são necessários os
anos **2020 a 2025**:

### 3. Treinar e gerar os gráficos

```bash
python treino_svr.py     # SVR — mais rápido
python treino_rna.py     # RNA — requer TensorFlow; mais lento
```

## Saídas

| Arquivo / pasta                          | Conteúdo                                                        |
|------------------------------------------|----------------------------------------------------------------|
| `graficos_rna/teste_XXX_rna.png`         | Dispersão real vs. previsto por cenário (RNA)                  |
| `graficos_svr/teste_XXX_svr.png`         | Dispersão real vs. previsto por cenário (SVR)                  |
| `resumo_resultados_rna.csv`              | Métricas por cenário (RNA)                                     |
| `resumo_resultados_svr.csv`              | Métricas por cenário (SVR)                                     |
| `historico/curva_aprendizado_teste_XXX.png` | Curva de perda e R² por época (cenários de dezembro, RNA)  |
| `historico/historico_teste_XXX.npz`      | Histórico bruto de treino para replotagem (dezembro, RNA)     |

> A cada execução, o CSV é **recriado do zero** e o número do teste é fixo
> (`Teste 1`..`Teste 18`, igual ao índice do cenário). Rodar a bateria de novo
> reproduz exatamente a mesma numeração e sobrescreve os gráficos
> (`teste_XXX_*.png`) — não acumula entre rodadas.

## Cenários

6 períodos de teste × 3 janelas de treino:

- **Períodos:** três dias isolados de junho/2025 (sexta 06, sábado 07,
  domingo 08) e três meses inteiros (junho, setembro e dezembro de 2025).
- **Janelas de treino:** 1 ano, 3 anos e 5 anos anteriores ao início do teste.

## Notas

- **Hiperparâmetros:** fixos por cenário, definidos diretamente em
  `BATERIA_TESTES` em cada script (os mesmos usados no trabalho). Para alterar
  um cenário, edite os campos correspondentes (`dropout`, `learning_rate`,
  `batch_size`, `huber_delta` na RNA; `C`, `gamma`, `epsilon` no SVR).
- **RNA:** alvo em escala log (`log1p`), perda de Huber, *early stopping* e
  `ReduceLROnPlateau`. A curva de aprendizado (perda e R² por época) é gerada
  apenas nos cenários de dezembro, para não onerar os demais.
- **SVR:** kernel RBF, alvo em escala log, com *lags* adicionais (48h, 168h),
  sazonalidade anual e indicador de fim de semana como features.
- Semente fixa (`SEED = 42`) para reprodutibilidade.
