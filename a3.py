"""Pipeline de preparação, modelagem e avaliação do IDEB.

O pipeline combina a preparação observada nos notebooks do colega com a
arquitetura original do modelo: divisão 80/20, baseline de média simples e
Random Forest Regressor.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
CAMINHO_DADOS = BASE_DIR / "base_treino_model.csv"
CAMINHO_DESPESA = BASE_DIR / "investimento_educacao.csv"
COLUNAS_CATEGORICAS = ["sigla_partido", "sigla_uf_x"]
COLUNAS_NUMERICAS = ["ano", "despesa_educacao"]
COLUNAS_MODELO = COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS
COLUNA_ALVO = "ideb"


def carregar_e_cruzar_dados(caminho: Path, caminho_despesa: Path) -> pd.DataFrame:
    """Carrega a base do IDEB e cruza a despesa por município e ano."""
    df = pd.read_csv(caminho)
    df_despesa = pd.read_csv(caminho_despesa)

    colunas_despesa = {"id_municipio", "ano"}
    if "despesa_educacao" not in df_despesa.columns:
        if "investimento_educacao" in df_despesa.columns:
            df_despesa = df_despesa.rename(
                columns={"investimento_educacao": "despesa_educacao"}
            )
        else:
            colunas_despesa.add("despesa_educacao")

    ausentes = colunas_despesa.difference(df_despesa.columns)
    if ausentes:
        raise ValueError(
            "Colunas necessárias ausentes no CSV de despesa: "
            + ", ".join(sorted(ausentes))
        )

    for frame in (df, df_despesa):
        frame["id_municipio"] = pd.to_numeric(
            frame["id_municipio"], errors="coerce"
        ).astype("Int64")
        frame["ano"] = pd.to_numeric(frame["ano"], errors="coerce").astype("Int64")

    df_despesa["despesa_educacao"] = pd.to_numeric(
        df_despesa["despesa_educacao"], errors="coerce"
    )
    if df_despesa.duplicated(["id_municipio", "ano"]).any():
        raise ValueError(
            "O CSV de despesa contém chaves id_municipio/ano duplicadas."
        )

    df = df.merge(
        df_despesa[["id_municipio", "ano", "despesa_educacao"]],
        on=["id_municipio", "ano"],
        how="left",
        validate="many_to_one",
    )
    correspondencias = df["despesa_educacao"].notna().mean() * 100
    print(
        f"\nCorrespondência da despesa com o IDEB: "
        f"{correspondencias:.2f}% das linhas"
    )
    return df


def preparar_dados(caminho: Path, caminho_despesa: Path) -> pd.DataFrame:
    """Carrega, limpa e resume os dados antes da etapa de modelagem.

    As conversões explícitas e o tratamento de ausências seguem o padrão dos
    scripts do colega, que preservam identificadores/categorias como texto e
    convertem valores numéricos antes da análise.
    """
    df = carregar_e_cruzar_dados(caminho, caminho_despesa)
    df.columns = df.columns.str.strip()

    colunas_necessarias = set(COLUNAS_MODELO + [COLUNA_ALVO])
    ausentes = colunas_necessarias.difference(df.columns)
    if ausentes:
        raise ValueError(
            "Colunas necessárias ausentes no arquivo: "
            + ", ".join(sorted(ausentes))
        )

    # Remove linhas repetidas e normaliza as categorias textuais.
    df = df.drop_duplicates().copy()
    for coluna in COLUNAS_CATEGORICAS:
        df[coluna] = df[coluna].astype("string").str.strip().str.upper()
        df[coluna] = df[coluna].replace({"": pd.NA, "NAN": pd.NA})

    # Converte os campos numéricos, transformando valores inválidos em ausentes.
    for coluna in COLUNAS_NUMERICAS + [COLUNA_ALVO]:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    # O alvo não pode ser imputado: essas linhas não permitem avaliar o modelo.
    df = df.dropna(subset=[COLUNA_ALVO]).copy()

    # Imputa preditores numéricos pela mediana e categóricos pela moda.
    for coluna in COLUNAS_NUMERICAS:
        df[coluna] = df[coluna].fillna(df[coluna].median())
    for coluna in COLUNAS_CATEGORICAS:
        moda = df[coluna].mode(dropna=True)
        df[coluna] = df[coluna].fillna(moda.iloc[0] if not moda.empty else "NA")

    if df.empty:
        raise ValueError("Não há dados válidos após a etapa de limpeza.")

    # EDA mínima e reproduzível: permite verificar a preparação no terminal.
    print("\n=== ANÁLISE EXPLORATÓRIA (EDA) ===")
    print(f"Dimensões após limpeza: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print("\nValores ausentes por coluna:")
    print(df[COLUNAS_MODELO + [COLUNA_ALVO]].isna().sum().to_string())
    print("\nResumo estatístico:")
    print(df[COLUNAS_NUMERICAS + [COLUNA_ALVO]].describe().round(3).to_string())

    return df


def calcular_metricas(y_real: pd.Series, y_pred: np.ndarray, nome: str) -> None:
    """Calcula e imprime as métricas solicitadas para um conjunto de previsões."""
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    print(f"\n--- {nome} ---")
    print(f"RMSE (Erro Quadrático Médio): {rmse:.4f}")
    print(f"MAE (Erro Absoluto Médio): {mae:.4f}")
    print(f"R² (Coeficiente de Determinação): {r2:.4f}")


def main() -> None:
    """Executa as fases de preparação, modelagem e resultados."""
    # FASE 1 — preparação e EDA.
    df = preparar_dados(CAMINHO_DADOS, CAMINHO_DESPESA)

    # A seleção ocorre depois da limpeza; X e y são as entradas do modelo.
    # get_dummies mantém o One-Hot Encoding da implementação original.
    X = pd.get_dummies(df[COLUNAS_MODELO], columns=COLUNAS_CATEGORICAS, drop_first=True)
    y = df[COLUNA_ALVO]

    # FASE 2 — divisão 80/20, baseline e Random Forest.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred_baseline = np.full(len(y_test), y_train.mean(), dtype=float)

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred_model = model.predict(X_test)

    print("\n=== RESULTADOS DE AVALIAÇÃO DO MODELO ===")
    calcular_metricas(y_test, y_pred_baseline, "BASELINE (Média Simples)")
    calcular_metricas(y_test, y_pred_model, "RANDOM FOREST REGRESSOR")

    # Regressão Linear Múltipla com escalonamento exclusivo deste modelo.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    y_pred_lr = model_lr.predict(X_test_scaled)
    calcular_metricas(y_test, y_pred_lr, "REGRESSÃO LINEAR MÚLTIPLA")

    # FASE 3 — importância das variáveis e gráficos.
    df_importancia = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\n=== TOP 15 VARIÁVEIS MAIS IMPORTANTES ===")
    print(df_importancia.head(15).to_string(index=False))

    plt.figure(figsize=(10, 6))
    top_importances = df_importancia.head(15).sort_values("importance")
    plt.barh(top_importances["feature"], top_importances["importance"])
    plt.xlabel("Importância")
    plt.ylabel("Variável")
    plt.title("Feature Importance — Random Forest")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "grafico_feature_importance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_model, alpha=0.6)
    limites = [min(y_test.min(), y_pred_model.min()), max(y_test.max(), y_pred_model.max())]
    plt.plot(limites, limites, "r--", label="Previsão ideal")
    plt.xlabel("Valor real do IDEB")
    plt.ylabel("Valor previsto do IDEB")
    plt.title("Valores reais vs. previstos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "grafico_real_vs_previsto.png", dpi=150)
    plt.close()

    print("\nGráficos exportados com sucesso.")


if __name__ == "__main__":
    main()
