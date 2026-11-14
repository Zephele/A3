"""Pipeline de preparação, modelagem e avaliação do IDEB municipal.

Avalia o desempenho preditivo em corte temporal (treino <= 2019, teste >= 2021)
comparando baselines, Regressão Linear Múltipla e Random Forest Regressor.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
CAMINHO_DADOS = BASE_DIR / "base_treino_model.csv"
CAMINHO_DESPESA = BASE_DIR / "investimento_educacao.csv"
ANO_CORTE_TREINO = 2019
ANO_INICIO_TESTE = 2021
COLUNA_ALVO = "ideb"


def carregar_e_cruzar_dados(
    caminho: Path, caminho_despesa: Path
) -> pd.DataFrame:
    """Carrega o IDEB e cruza a despesa real por município e ano."""
    df = pd.read_csv(caminho)
    df_despesa = pd.read_csv(caminho_despesa)
    df.columns = df.columns.str.strip()
    df_despesa.columns = df_despesa.columns.str.strip()

    if "sigla_uf" not in df.columns and "sigla_uf_x" in df.columns:
        df = df.rename(columns={"sigla_uf_x": "sigla_uf"})
    if (
        "despesa_educacao" not in df_despesa.columns
        and "investimento_educacao" in df_despesa.columns
    ):
        df_despesa = df_despesa.rename(
            columns={"investimento_educacao": "despesa_educacao"}
        )

    ausentes = {"id_municipio", "ano", "despesa_educacao"}.difference(
        df_despesa.columns
    )
    if ausentes:
        raise ValueError(
            "Colunas ausentes no CSV de despesa: "
            + ", ".join(sorted(ausentes))
        )

    for frame in (df, df_despesa):
        frame["id_municipio"] = pd.to_numeric(
            frame["id_municipio"], errors="coerce"
        ).astype("Int64")
        frame["ano"] = pd.to_numeric(frame["ano"], errors="coerce").astype("Int64")

    df = df[df["id_municipio"] >= 1_100_000].copy()
    df_despesa = df_despesa[df_despesa["id_municipio"] >= 1_100_000].copy()
    df_despesa["despesa_educacao"] = pd.to_numeric(
        df_despesa["despesa_educacao"], errors="coerce"
    )
    df_despesa = df_despesa.dropna(
        subset=["id_municipio", "ano", "despesa_educacao"]
    )
    df_despesa = df_despesa[df_despesa["despesa_educacao"] > 0].drop_duplicates(
        ["id_municipio", "ano"]
    )

    df = df.merge(
        df_despesa[["id_municipio", "ano", "despesa_educacao"]],
        on=["id_municipio", "ano"],
        how="left",
        validate="many_to_one",
    )
    correspondencia = df["despesa_educacao"].notna().mean() * 100
    print(f"Correspondência bruta da despesa com o IDEB: {correspondencia:.2f}%")
    return df


def preparar_dados(
    caminho: Path, caminho_despesa: Path
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Limpa os dados, calcula o IDEB anterior e evita imputação artificial."""
    df = carregar_e_cruzar_dados(caminho, caminho_despesa)
    colunas_categoricas = ["sigla_partido", "sigla_uf"]
    for extra in ("ensino", "anos_escolares"):
        if extra in df.columns:
            colunas_categoricas.append(extra)

    if "anos_escolares" not in df.columns:
        print(
            "[AVISO] Coluna 'anos_escolares' ausente. Execute 'preprocess.py' "
            "para evitar mistura entre etapas."
        )

    for coluna in colunas_categoricas:
        df[coluna] = df[coluna].astype("string").str.strip().str.upper()
        df[coluna] = df[coluna].replace(
            {"": pd.NA, "NAN": pd.NA, "NONE": pd.NA}
        )

    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["despesa_educacao"] = pd.to_numeric(
        df["despesa_educacao"], errors="coerce"
    )
    df[COLUNA_ALVO] = pd.to_numeric(df[COLUNA_ALVO], errors="coerce")
    df = df.dropna(
        subset=["id_municipio", "ano", COLUNA_ALVO] + colunas_categoricas
    ).copy()

    chave_painel = ["id_municipio"] + [
        coluna
        for coluna in ("ensino", "anos_escolares")
        if coluna in df.columns
    ] + ["ano"]
    df = df.drop_duplicates(subset=chave_painel).sort_values(chave_painel)
    df = df.reset_index(drop=True)

    grupo_lag = ["id_municipio"] + [
        coluna
        for coluna in ("ensino", "anos_escolares")
        if coluna in df.columns
    ]
    df["ideb_anterior"] = df.groupby(grupo_lag)[COLUNA_ALVO].shift(1)

    df = df[df["ano"] >= 2013].dropna(subset=["despesa_educacao"]).copy()
    limite_superior = df["despesa_educacao"].quantile(0.995)
    df = df[
        (df["despesa_educacao"] > 0)
        & (df["despesa_educacao"] <= limite_superior)
    ].copy()
    df["log_despesa_educacao"] = np.log1p(df["despesa_educacao"])

    if df.empty:
        raise ValueError(
            "Não há dados válidos após a limpeza e o filtro temporal (>= 2013)."
        )

    colunas_numericas = ["ano", "log_despesa_educacao"]
    print("\n=== ANÁLISE EXPLORATÓRIA (PERÍODO 2013+) ===")
    print(f"Dimensões após limpeza: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"Anos presentes: {sorted(df['ano'].unique().tolist())}")
    print(
        df[
            ["ano", "despesa_educacao", "log_despesa_educacao", COLUNA_ALVO]
        ].describe().round(3).to_string()
    )
    return df, colunas_categoricas, colunas_numericas


def calcular_metricas(
    y_real: pd.Series, y_pred: np.ndarray, nome: str
) -> dict[str, float]:
    """Calcula e exibe RMSE, MAE e R²."""
    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    mae = float(mean_absolute_error(y_real, y_pred))
    r2 = float(r2_score(y_real, y_pred))
    print(f"\n--- {nome} ---")
    print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")
    print(f"MAE (Erro Absoluto Médio):            {mae:.4f}")
    print(f"R² (Coeficiente de Determinação):     {r2:.4f}")
    return {"modelo": nome, "rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    """Executa preparação, modelagem, avaliação e exportação dos gráficos."""
    df, colunas_categoricas, colunas_numericas = preparar_dados(
        CAMINHO_DADOS, CAMINHO_DESPESA
    )
    colunas_modelo = colunas_categoricas + colunas_numericas
    x_all = pd.get_dummies(
        df[colunas_modelo], columns=colunas_categoricas, drop_first=True
    )
    y_all = df[COLUNA_ALVO]

    mask_treino = df["ano"] <= ANO_CORTE_TREINO
    mask_teste = df["ano"] >= ANO_INICIO_TESTE
    x_train, y_train = x_all[mask_treino], y_all[mask_treino]
    x_test, y_test = x_all[mask_teste], y_all[mask_teste]
    grupos_treino = df.loc[mask_treino, "id_municipio"]
    if x_train.empty or x_test.empty:
        raise ValueError("O corte temporal não produziu treino e teste não vazios.")

    print(
        f"\nDivisão temporal -> treino (<= {ANO_CORTE_TREINO}): {len(x_train)} | "
        f"teste (>= {ANO_INICIO_TESTE}): {len(x_test)}"
    )
    y_pred_media = np.full(len(y_test), y_train.mean(), dtype=float)
    y_pred_persistencia = (
        df.loc[mask_teste, "ideb_anterior"].fillna(y_train.mean()).to_numpy()
    )

    espaco_hiperparametros = {
        "n_estimators": [150, 250],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [5, 10, 20],
        "max_features": ["sqrt", 0.5, 0.8],
    }
    busca_rf = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=espaco_hiperparametros,
        n_iter=10,
        cv=GroupKFold(n_splits=3),
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    busca_rf.fit(x_train, y_train, groups=grupos_treino)
    model_rf = busca_rf.best_estimator_
    y_pred_rf = model_rf.predict(x_test)
    print(f"Melhores hiperparâmetros: {busca_rf.best_params_}")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model_lr = LinearRegression().fit(x_train_scaled, y_train)
    y_pred_lr = model_lr.predict(x_test_scaled)

    print("\n=== RESULTADOS NA AVALIAÇÃO TEMPORAL (>= 2021) ===")
    calcular_metricas(y_test, y_pred_media, "BASELINE 1 — Média Simples")
    calcular_metricas(
        y_test,
        y_pred_persistencia,
        "BASELINE 2 — Persistência (IDEB anterior)",
    )
    calcular_metricas(y_test, y_pred_lr, "REGRESSÃO LINEAR MÚLTIPLA")
    calcular_metricas(y_test, y_pred_rf, "RANDOM FOREST REGRESSOR (Ajustado)")

    x_tr_rand, x_te_rand, y_tr_rand, y_te_rand = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42
    )
    rf_diagnostico = RandomForestRegressor(
        **busca_rf.best_params_, random_state=42, n_jobs=-1
    )
    rf_diagnostico.fit(x_tr_rand, y_tr_rand)
    print("\n=== DIAGNÓSTICO DE VAZAMENTO (SPLIT ALEATÓRIO 80/20) ===")
    calcular_metricas(
        y_te_rand,
        rf_diagnostico.predict(x_te_rand),
        "RANDOM FOREST (Split Aleatório)",
    )

    df_importancia = pd.DataFrame(
        {"feature": x_all.columns, "importance": model_rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    def agrupar_variavel(nome_coluna: str) -> str:
        for categoria in colunas_categoricas:
            if nome_coluna.startswith(f"{categoria}_"):
                return categoria
        return nome_coluna

    df_importancia["grupo"] = df_importancia["feature"].apply(agrupar_variavel)
    resumo_grupos = (
        df_importancia.groupby("grupo")["importance"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    resumo_grupos["importancia_%"] = (resumo_grupos["importance"] * 100).round(2)
    print("\n=== IMPORTÂNCIA AGREGADA POR GRUPO ===")
    print(resumo_grupos[["grupo", "importancia_%"]].to_string(index=False))

    plt.figure(figsize=(10, 6))
    top_importances = df_importancia.head(15).sort_values("importance")
    plt.barh(top_importances["feature"], top_importances["importance"])
    plt.xlabel("Importância relativa")
    plt.ylabel("Variável")
    plt.title("Top 15 Variáveis — Random Forest (Corte Temporal)")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "grafico_feature_importance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.35, s=15)
    limites = [min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())]
    plt.plot(limites, limites, "r--", label="Previsão ideal")
    plt.xlabel("Valor real do IDEB (>= 2021)")
    plt.ylabel("Valor previsto do IDEB")
    plt.title("IDEB Real vs. Previsto — Avaliação Temporal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "grafico_real_vs_previsto.png", dpi=150)
    plt.close()
    print("\nGráficos exportados com sucesso.")


if __name__ == "__main__":
    main()
