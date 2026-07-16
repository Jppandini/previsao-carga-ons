# ============================================
# Loader da Curva de Carga Horária do ONS
# Fonte: https://dados.ons.org.br/dataset/curva-carga
# ============================================
#
# Substitui a origem BigQuery dos scripts de treino por leitura direta dos
# arquivos parquet anuais do ONS (CURVA_CARGA_<ANO>.parquet).
#
# O dataset traz a carga horaria por subsistema (N, NE, S, SE/CO). A carga
# total do SIN e obtida somando os subsistemas em cada instante horario, o que
# reproduz o "SUM(...) GROUP BY hora" da query BigQuery original.

import glob
import os

import numpy as np
import pandas as pd

# Nome esperado da coluna de data/hora no dataset do ONS.
COLUNA_INSTANTE = "din_instante"


def _detectar_coluna_valor(colunas):
    """Encontra a coluna de carga (val_carga...) sem depender do nome exato.

    O ONS ja publicou variacoes do nome (val_cargaenergiamwmed e, em versoes
    antigas, val_cargaenergiahomwmed). Detectar por prefixo evita quebrar caso
    o nome mude entre anos.
    """
    candidatas = [c for c in colunas if c.lower().startswith("val_carga")]
    if not candidatas:
        raise ValueError(
            f"Nenhuma coluna de carga (val_carga*) encontrada. Colunas: {list(colunas)}"
        )
    return candidatas[0]


def _detectar_coluna_instante(colunas):
    if COLUNA_INSTANTE in colunas:
        return COLUNA_INSTANTE
    candidatas = [c for c in colunas if c.lower().startswith("din_")]
    if not candidatas:
        raise ValueError(
            f"Nenhuma coluna de instante (din_*) encontrada. Colunas: {list(colunas)}"
        )
    return candidatas[0]


def carregar_curva_carga(pasta_dados="dados", verbose=True):
    """Le os parquet do ONS em `pasta_dados` e retorna a carga total horaria.

    Retorna um DataFrame com as mesmas colunas que a query BigQuery original
    entregava, para manter o restante dos scripts de treino inalterado:

        dia, mes, ano, hora, val_geracao

    Cada linha e a soma da carga (MWmed) de todos os subsistemas naquele
    instante horario, ordenada por ano, mes, dia, hora.
    """
    padrao = os.path.join(pasta_dados, "*.parquet")
    arquivos = sorted(glob.glob(padrao))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum parquet encontrado em '{padrao}'.\n"
            "Baixe os arquivos CURVA_CARGA_<ANO>.parquet do ONS para essa pasta.\n"
            "Ex.: https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/"
            "curva-carga-ho/CURVA_CARGA_2025.parquet"
        )

    if verbose:
        print(f"🔄 Lendo {len(arquivos)} arquivo(s) parquet de '{pasta_dados}'...")

    partes = []
    for caminho in arquivos:
        df = pd.read_parquet(caminho)
        col_instante = _detectar_coluna_instante(df.columns)
        col_valor = _detectar_coluna_valor(df.columns)

        parte = pd.DataFrame({
            "din_instante": pd.to_datetime(df[col_instante]),
            "val_carga": pd.to_numeric(df[col_valor], errors="coerce"),
        })
        partes.append(parte)
        if verbose:
            print(f"   ✅ {os.path.basename(caminho)}: {len(parte)} registros")

    bruto = pd.concat(partes, ignore_index=True)

    # Replica o filtro da query original: descarta cargas nao positivas/invalidas.
    bruto = bruto.dropna(subset=["din_instante", "val_carga"])
    bruto = bruto[bruto["val_carga"] > 0]

    # Trunca para a hora e soma os subsistemas -> carga total do SIN por hora.
    bruto["instante_hora"] = bruto["din_instante"].dt.floor("h")
    agregado = (
        bruto.groupby("instante_hora", as_index=False)["val_carga"]
        .sum()
        .rename(columns={"val_carga": "val_geracao"})
    )
    agregado["val_geracao"] = agregado["val_geracao"].round(3)

    agregado["dia"] = agregado["instante_hora"].dt.day
    agregado["mes"] = agregado["instante_hora"].dt.month
    agregado["ano"] = agregado["instante_hora"].dt.year
    agregado["hora"] = agregado["instante_hora"].dt.hour

    resultado = (
        agregado[["dia", "mes", "ano", "hora", "val_geracao"]]
        .sort_values(["ano", "mes", "dia", "hora"])
        .reset_index(drop=True)
    )

    if verbose:
        inicio = resultado[["ano", "mes", "dia"]].iloc[0].tolist()
        fim = resultado[["ano", "mes", "dia"]].iloc[-1].tolist()
        print(
            f"✅ Carga total horaria: {len(resultado)} registros "
            f"({inicio[2]:02d}/{inicio[1]:02d}/{inicio[0]} a "
            f"{fim[2]:02d}/{fim[1]:02d}/{fim[0]})"
        )

    return resultado


if __name__ == "__main__":
    # Inspecao rapida: python ons_data.py [pasta_dados]
    import sys

    pasta = sys.argv[1] if len(sys.argv) > 1 else "dados"
    df = carregar_curva_carga(pasta)
    print(df.head())
    print(df.describe())
