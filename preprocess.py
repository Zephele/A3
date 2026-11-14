"""Pré-processamento e cruzamento das bases públicas do TSE e do INEP."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CAMINHO_PREFEITOS = BASE_DIR / "bq-results.csv"
CAMINHO_IDEB = BASE_DIR / "br_inep_ideb_municipio.csv"
CAMINHO_INTERMEDIARIO = BASE_DIR / "dataset_processado_a3.csv"
CAMINHO_SAIDA = BASE_DIR / "base_treino_model.csv"


def main() -> None:
    """Gera as bases intermediária e final usadas pelo modelo."""
    df_prefeitos = pd.read_csv(CAMINHO_PREFEITOS)
    df_ideb = pd.read_csv(CAMINHO_IDEB)
    df_prefeitos["id_municipio"] = df_prefeitos["id_municipio"].astype(str).str.strip()
    df_ideb["id_municipio"] = df_ideb["id_municipio"].astype(str).str.strip()

    cargo_prefeito = (
        df_prefeitos["cargo"].astype(str).str.lower().str.strip() == "prefeito"
    )
    resultado = df_prefeitos["resultado"].astype(str).str.lower().str.strip()
    eleito = resultado.str.contains("eleito", na=False) & ~resultado.str.contains(
        r"n[aã]o eleito", regex=True, na=False
    )
    df_eleitos = df_prefeitos[cargo_prefeito & eleito].copy()

    mandatos = []
    for row in df_eleitos.itertuples(index=False):
        for ano_mandato in range(int(row.ano) + 1, int(row.ano) + 5):
            mandatos.append(
                {
                    "id_municipio": row.id_municipio,
                    "sigla_uf": row.sigla_uf,
                    "ano": ano_mandato,
                    "sigla_partido": row.sigla_partido,
                }
            )
    df_mandatos = pd.DataFrame(mandatos).drop_duplicates(
        subset=["id_municipio", "ano"]
    )

    df_ideb_limpo = df_ideb.drop(columns=["sigla_uf"], errors="ignore")
    df_merged = pd.merge(
        df_mandatos,
        df_ideb_limpo,
        on=["id_municipio", "ano"],
        how="inner",
    )
    df_merged.to_csv(CAMINHO_INTERMEDIARIO, index=False)
    print(f"Base cruzada salva em '{CAMINHO_INTERMEDIARIO.name}' ({len(df_merged)} linhas).")

    rede_municipal = (
        df_merged["rede"].astype(str).str.lower().str.strip() == "municipal"
    )
    df_clean = df_merged[rede_municipal & df_merged["ideb"].notna()].copy()
    colunas_desejadas = [
        "ano",
        "sigla_uf",
        "id_municipio",
        "sigla_partido",
        "ensino",
        "anos_escolares",
        "taxa_aprovacao",
        "ideb",
    ]
    colunas_presentes = [coluna for coluna in colunas_desejadas if coluna in df_clean]
    chave_unica = [
        coluna
        for coluna in ["id_municipio", "ano", "ensino", "anos_escolares"]
        if coluna in colunas_presentes
    ]
    df_model = (
        df_clean[colunas_presentes]
        .dropna(
            subset=[
                coluna
                for coluna in [
                    "ano",
                    "sigla_uf",
                    "id_municipio",
                    "sigla_partido",
                    "ideb",
                ]
                if coluna in colunas_presentes
            ]
        )
        .drop_duplicates(subset=chave_unica)
        .sort_values(chave_unica)
    )
    df_model.to_csv(CAMINHO_SAIDA, index=False)
    print(f"Base final salva em '{CAMINHO_SAIDA.name}' ({len(df_model)} linhas).")


if __name__ == "__main__":
    main()
