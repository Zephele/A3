"""Pipeline de modelagem e avaliação do IDEB municipal (A3).

Pergunta: o partido do prefeito ajuda a prever o IDEB do município *além* do
que já é explicado pelo histórico do próprio município (inércia), pela UF e
pela despesa em educação?

Desenho do experimento
----------------------
* Avaliação temporal: treino com anos <= 2019, teste com anos >= 2021.
* O IDEB é medido a cada 2 anos; ``ideb_anterior`` é o IDEB do mesmo município
  e da mesma etapa dois anos antes (sem preenchimento artificial).
* Baselines: média do treino e persistência (prever o IDEB anterior).
* Dois alvos: ``nivel`` (IDEB) e ``variacao`` (IDEB - IDEB anterior). Em
  ``variacao`` o modelo prevê a mudança e soma ao IDEB anterior; as métricas
  são sempre calculadas na escala do IDEB.
* Ablação em blocos (cada bloco inclui o anterior):
    A  inércia   = ideb_anterior + ano + etapa
    B  + região  = A + UF
    C  + despesa = B + log(despesa em educação)
    D  + partido = C + partido do prefeito
  A contribuição do partido é a diferença entre D e C, com intervalo de
  confiança por bootstrap em blocos de município.
* Ajuste de hiperparâmetros do Random Forest com validação de janela
  expansiva (respeita a ordem do tempo), nunca com o conjunto de teste.

Limitações conhecidas (para o relatório)
----------------------------------------
* O partido é o do prefeito eleito; trocas de partido, cassações e mandatos-
  tampão não são modeladas.
* A despesa é nominal (não deflacionada) e absoluta (não per capita/aluno).
* O período de teste inclui a pandemia (SAEB 2021).
* Importância de variáveis e ablação medem associação preditiva, não
  causalidade.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
CAMINHO_DADOS = BASE_DIR / "base_treino_model.csv"
CAMINHO_DESPESA = BASE_DIR / "investimento_educacao.csv"
CAMINHO_RESULTADOS = BASE_DIR / "resultados_modelos.csv"

ANO_CORTE_TREINO = 2019
ANO_INICIO_TESTE = 2021
ANO_INICIO_DESPESA = 2013  # a despesa só tem cobertura real a partir daqui
PASSO_IDEB = 2  # o IDEB é divulgado a cada dois anos
COLUNA_ALVO = "ideb"
MODO_PRINCIPAL = "variacao"  # alinhado à pergunta ("evolução" do indicador)
N_BOOTSTRAP = 2000
LIMIAR_RELEVANCIA = 0.01  # diferenças de MAE menores que isso (em pontos de IDEB) são desprezíveis

# Janelas expansivas para o ajuste de hiperparâmetros: (treino até, valida em).
JANELAS_CV = [(2015, 2017), (2017, 2019)]

ESPACO_HIPERPARAMETROS = {
    "modelo__n_estimators": [150, 250],
    "modelo__max_depth": [8, 12, 16, None],
    "modelo__min_samples_leaf": [5, 10, 20],
    "modelo__max_features": ["sqrt", 0.5, 0.8],
}


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def carregar_e_cruzar_dados(caminho: Path, caminho_despesa: Path) -> pd.DataFrame:
    """Carrega o IDEB (já com partido) e cruza a despesa por município e ano."""
    df = pd.read_csv(caminho)
    df_despesa = pd.read_csv(caminho_despesa)
    df.columns = df.columns.str.strip()
    df_despesa.columns = df_despesa.columns.str.strip()

    if "sigla_uf" not in df.columns and "sigla_uf_x" in df.columns:
        df = df.rename(columns={"sigla_uf_x": "sigla_uf"})
    if "despesa_educacao" not in df_despesa.columns:
        df_despesa = df_despesa.rename(
            columns={"investimento_educacao": "despesa_educacao"}
        )

    ausentes = {"id_municipio", "ano", "despesa_educacao"} - set(df_despesa.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes no CSV de despesa: {sorted(ausentes)}")

    for frame in (df, df_despesa):
        frame["id_municipio"] = pd.to_numeric(
            frame["id_municipio"], errors="coerce"
        ).astype("Int64")
        frame["ano"] = pd.to_numeric(frame["ano"], errors="coerce").astype("Int64")

    # Códigos IBGE de município válidos começam em 11xxxxx.
    df = df[df["id_municipio"] >= 1_100_000].copy()
    df_despesa = df_despesa[df_despesa["id_municipio"] >= 1_100_000].copy()
    df_despesa["despesa_educacao"] = pd.to_numeric(
        df_despesa["despesa_educacao"], errors="coerce"
    )
    df_despesa = df_despesa.dropna(subset=["id_municipio", "ano", "despesa_educacao"])
    df_despesa = df_despesa[df_despesa["despesa_educacao"] > 0].drop_duplicates(
        ["id_municipio", "ano"]
    )

    return df.merge(
        df_despesa[["id_municipio", "ano", "despesa_educacao"]],
        on=["id_municipio", "ano"],
        how="left",
        validate="many_to_one",
    )


def adicionar_ideb_anterior(df: pd.DataFrame, chave: list[str]) -> pd.DataFrame:
    """Anexa o IDEB medido ``PASSO_IDEB`` anos antes (mesmo município e etapa).

    Usa junção por ano (e não ``shift``) para nunca puxar uma medição de um
    ano diferente do esperado quando há lacunas no painel.
    """
    anterior = df[chave + ["ano", COLUNA_ALVO]].copy()
    anterior["ano"] = anterior["ano"] + PASSO_IDEB
    anterior = anterior.rename(columns={COLUNA_ALVO: "ideb_anterior"})
    return df.merge(anterior, on=chave + ["ano"], how="left", validate="one_to_one")


def preparar_dados(caminho: Path, caminho_despesa: Path) -> pd.DataFrame:
    """Limpa, cria o IDEB anterior e a variação, e restringe ao período com despesa."""
    df = carregar_e_cruzar_dados(caminho, caminho_despesa)

    colunas_cat = ["sigla_partido", "sigla_uf"]
    colunas_etapa = [c for c in ("ensino", "anos_escolares") if c in df.columns]
    if "anos_escolares" not in colunas_etapa:
        raise ValueError(
            "Coluna 'anos_escolares' ausente: execute 'python preprocess.py' antes."
        )

    for coluna in colunas_cat + colunas_etapa:
        df[coluna] = df[coluna].astype("string").str.strip().str.upper()
        df[coluna] = df[coluna].replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})

    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["despesa_educacao"] = pd.to_numeric(df["despesa_educacao"], errors="coerce")
    df[COLUNA_ALVO] = pd.to_numeric(df[COLUNA_ALVO], errors="coerce")
    df = df.dropna(
        subset=["id_municipio", "ano", COLUNA_ALVO] + colunas_cat + colunas_etapa
    )

    chave = ["id_municipio"] + colunas_etapa
    df = df.drop_duplicates(subset=chave + ["ano"]).reset_index(drop=True)
    df = adicionar_ideb_anterior(df, chave)  # antes de cortar o período
    df["variacao_ideb"] = df[COLUNA_ALVO] - df["ideb_anterior"]

    df = df[df["ano"] >= ANO_INICIO_DESPESA].dropna(subset=["despesa_educacao"])
    df = df.dropna(subset=["ideb_anterior"]).copy()

    # Limite superior da despesa estimado só no treino (evita vazamento) e
    # aplicado por truncamento, sem descartar linhas do teste.
    treino = df["ano"] <= ANO_CORTE_TREINO
    limite = df.loc[treino, "despesa_educacao"].quantile(0.995)
    df["log_despesa_educacao"] = np.log1p(df["despesa_educacao"].clip(upper=limite))

    # Colunas constantes não informam nada (ex.: 'ensino' só tem 'fundamental').
    constantes = [c for c in colunas_etapa if df[c].nunique() < 2]
    if constantes:
        print(f"[INFO] Colunas constantes ignoradas: {constantes}")
    df.attrs["colunas_etapa"] = [c for c in colunas_etapa if c not in constantes]

    print("=== ANÁLISE EXPLORATÓRIA ===")
    print(f"Linhas com IDEB atual, IDEB anterior e despesa: {len(df)}")
    print(f"Municípios: {df['id_municipio'].nunique()} | partidos: "
          f"{df['sigla_partido'].nunique()} | UFs: {df['sigla_uf'].nunique()}")
    print(f"Anos: {sorted(df['ano'].unique().tolist())}")
    print(df[["ideb", "ideb_anterior", "variacao_ideb", "despesa_educacao"]]
          .describe().round(3).to_string())
    print("\nVariação média do IDEB por ano:")
    print(df.groupby("ano")["variacao_ideb"].agg(["mean", "std", "count"])
          .round(3).to_string())
    return df


# --------------------------------------------------------------------------
# Modelagem
# --------------------------------------------------------------------------
def definir_blocos(colunas_etapa: list[str]) -> dict[str, dict[str, list[str]]]:
    """Blocos cumulativos da ablação: cada um lista colunas numéricas/categóricas."""
    a = {"num": ["ideb_anterior", "ano"], "cat": list(colunas_etapa)}
    b = {"num": a["num"], "cat": a["cat"] + ["sigla_uf"]}
    c = {"num": b["num"] + ["log_despesa_educacao"], "cat": b["cat"]}
    d = {"num": c["num"], "cat": c["cat"] + ["sigla_partido"]}
    return {
        "A inércia": a,
        "B + região (UF)": b,
        "C + despesa": c,
        "D + partido": d,
    }


def criar_pipeline(tipo: str, bloco: dict[str, list[str]], **params) -> Pipeline:
    """Pré-processamento + modelo; 'lr' padroniza as numéricas, 'rf' não precisa."""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if tipo == "lr":
        pre = ColumnTransformer(
            [("num", StandardScaler(), bloco["num"]), ("cat", ohe, bloco["cat"])]
        )
        modelo = LinearRegression()
    else:
        pre = ColumnTransformer(
            [("num", "passthrough", bloco["num"]), ("cat", ohe, bloco["cat"])]
        )
        modelo = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
    return Pipeline([("pre", pre), ("modelo", modelo)])


def coluna_alvo(modo: str) -> str:
    return "variacao_ideb" if modo == "variacao" else COLUNA_ALVO


def para_escala_ideb(pred: np.ndarray, base: pd.DataFrame, modo: str) -> np.ndarray:
    """Converte a previsão do alvo para a escala do IDEB."""
    return pred + base["ideb_anterior"].to_numpy() if modo == "variacao" else pred


def metricas(y_real, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "r2": float(r2_score(y_real, y_pred)),
    }


def ajustar_hiperparametros(df_treino: pd.DataFrame, bloco, modo: str) -> dict:
    """Busca aleatória com janelas expansivas (treina no passado, valida no futuro)."""
    anos = df_treino["ano"].to_numpy()
    cortes = [
        (np.where(anos <= ate)[0], np.where(anos == valida)[0])
        for ate, valida in JANELAS_CV
    ]
    colunas = bloco["num"] + bloco["cat"]
    busca = RandomizedSearchCV(
        criar_pipeline("rf", bloco),
        param_distributions=ESPACO_HIPERPARAMETROS,
        n_iter=10,
        cv=cortes,
        scoring="neg_mean_absolute_error",
        random_state=SEED,
        n_jobs=1,  # o paralelismo já está dentro do Random Forest
    )
    busca.fit(df_treino[colunas], df_treino[coluna_alvo(modo)])
    melhores = {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()}
    print(f"  [{modo}] melhores hiperparâmetros: {melhores}")
    return melhores


def intervalo_bootstrap(
    dif_erro: np.ndarray, municipios: np.ndarray
) -> tuple[float, float, float]:
    """IC 95% da média de ``dif_erro`` reamostrando municípios inteiros."""
    tabela = (
        pd.DataFrame({"m": municipios, "d": dif_erro})
        .groupby("m")["d"].agg(["sum", "count"])
    )
    somas, contagens = tabela["sum"].to_numpy(), tabela["count"].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(tabela), size=(N_BOOTSTRAP, len(tabela)))
    medias = somas[idx].sum(axis=1) / contagens[idx].sum(axis=1)
    return float(dif_erro.mean()), *np.percentile(medias, [2.5, 97.5])


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------
def main() -> None:
    df = preparar_dados(CAMINHO_DADOS, CAMINHO_DESPESA)
    blocos = definir_blocos(df.attrs["colunas_etapa"])

    treino = df[df["ano"] <= ANO_CORTE_TREINO].reset_index(drop=True)
    teste = df[df["ano"] >= ANO_INICIO_TESTE].reset_index(drop=True)
    if treino.empty or teste.empty:
        raise ValueError("O corte temporal não produziu treino e teste não vazios.")
    print(f"\nTreino (<= {ANO_CORTE_TREINO}): {len(treino)} | "
          f"Teste (>= {ANO_INICIO_TESTE}): {len(teste)}")

    y_teste = teste[COLUNA_ALVO]
    resultados: list[dict] = []
    erros_abs: dict[tuple[str, str, str], np.ndarray] = {}

    def registrar(nome: str, modo: str, bloco: str, y_pred: np.ndarray) -> None:
        resultados.append({"modelo": nome, "alvo": modo, "blocos": bloco,
                           **metricas(y_teste, y_pred)})
        erros_abs[(nome, modo, bloco)] = np.abs(y_teste.to_numpy() - y_pred)

    # Baselines.
    registrar("Baseline média", "-", "-", np.full(len(teste), treino[COLUNA_ALVO].mean()))
    registrar("Baseline persistência", "-", "-", teste["ideb_anterior"].to_numpy())

    # Ajuste de hiperparâmetros: uma vez por alvo, no bloco completo (D).
    print("\n=== AJUSTE DE HIPERPARÂMETROS (janela expansiva no treino) ===")
    bloco_completo = blocos["D + partido"]
    params = {m: ajustar_hiperparametros(treino, bloco_completo, m)
              for m in ("nivel", "variacao")}

    # Ablação: todos os blocos, dois modelos, dois alvos.
    modelos_ajustados: dict[tuple[str, str, str], Pipeline] = {}
    for modo in ("nivel", "variacao"):
        for nome_bloco, bloco in blocos.items():
            colunas = bloco["num"] + bloco["cat"]
            for tipo, rotulo in (("lr", "Regressão linear"), ("rf", "Random Forest")):
                pipe = criar_pipeline(tipo, bloco, **(params[modo] if tipo == "rf" else {}))
                pipe.fit(treino[colunas], treino[coluna_alvo(modo)])
                pred = para_escala_ideb(pipe.predict(teste[colunas]), teste, modo)
                registrar(rotulo, modo, nome_bloco, pred)
                modelos_ajustados[(rotulo, modo, nome_bloco)] = pipe

    tabela = pd.DataFrame(resultados)
    tabela.to_csv(CAMINHO_RESULTADOS, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("\n=== RESULTADOS NO TESTE TEMPORAL (escala do IDEB) ===")
        print(tabela.to_string(index=False))

    # Contribuição do partido: bloco D contra bloco C, com IC por município.
    print("\n=== CONTRIBUIÇÃO DO PARTIDO (D vs C; positivo = partido ajuda) ===")
    municipios = teste["id_municipio"].to_numpy()
    for rotulo in ("Random Forest", "Regressão linear"):
        for modo in ("nivel", "variacao"):
            dif = (erros_abs[(rotulo, modo, "C + despesa")]
                   - erros_abs[(rotulo, modo, "D + partido")])
            media, lo, hi = intervalo_bootstrap(dif, municipios)
            veredito = ("melhora" if lo > 0 else "piora" if hi < 0
                        else "sem efeito distinguível de zero")
            if max(abs(lo), abs(hi)) < LIMIAR_RELEVANCIA:
                veredito += f" (efeito desprezível: < {LIMIAR_RELEVANCIA} ponto de IDEB)"
            print(f"{rotulo:17s} | alvo={modo:8s} | ganho de MAE = {media:+.4f} "
                  f"(IC95% {lo:+.4f} a {hi:+.4f}) -> {veredito}")

    # Importância por permutação (variáveis originais, não dummies) no teste.
    print(f"\n=== IMPORTÂNCIA POR PERMUTAÇÃO — RF, alvo={MODO_PRINCIPAL}, bloco D ===")
    pipe_rf = modelos_ajustados[("Random Forest", MODO_PRINCIPAL, "D + partido")]
    colunas_d = bloco_completo["num"] + bloco_completo["cat"]
    perm = permutation_importance(
        pipe_rf, teste[colunas_d], teste[coluna_alvo(MODO_PRINCIPAL)],
        scoring="neg_mean_absolute_error", n_repeats=5, random_state=SEED, n_jobs=-1,
    )
    imp = (pd.DataFrame({"variavel": colunas_d, "aumento_mae": perm.importances_mean,
                         "desvio": perm.importances_std})
           .sort_values("aumento_mae", ascending=False))
    print(imp.round(4).to_string(index=False))

    # Diagnóstico de vazamento: split aleatório com o mesmo modelo.
    x_tr, x_te, y_tr, y_te = train_test_split(
        df[colunas_d], df[coluna_alvo(MODO_PRINCIPAL)], test_size=0.2, random_state=SEED
    )
    rf_diag = criar_pipeline("rf", bloco_completo, **params[MODO_PRINCIPAL]).fit(x_tr, y_tr)
    pred_diag = para_escala_ideb(rf_diag.predict(x_te), x_te, MODO_PRINCIPAL)
    real_diag = df.loc[x_te.index, COLUNA_ALVO]
    print("\n=== DIAGNÓSTICO DE VAZAMENTO (split aleatório 80/20, apenas diagnóstico) ===")
    print({k: round(v, 4) for k, v in metricas(real_diag, pred_diag).items()})

    # Gráficos.
    comp = tabela[(tabela["blocos"].isin(["-", "D + partido"]))
                  & (tabela["alvo"].isin(["-", MODO_PRINCIPAL]))].copy()
    comp["rotulo"] = comp["modelo"]
    plt.figure(figsize=(9, 4.5))
    plt.barh(comp["rotulo"], comp["mae"])
    plt.xlabel("MAE no teste (menor é melhor)")
    plt.title(f"Modelos completos (alvo: {MODO_PRINCIPAL}) vs baselines")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "grafico_comparacao_modelos.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.barh(imp["variavel"][::-1], imp["aumento_mae"][::-1],
             xerr=imp["desvio"][::-1])
    plt.xlabel("Aumento do MAE ao embaralhar a variável")
    plt.title("Importância por permutação (teste temporal)")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "grafico_feature_importance.png", dpi=150)
    plt.close()

    pred_rf = para_escala_ideb(pipe_rf.predict(teste[colunas_d]), teste, MODO_PRINCIPAL)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_teste, pred_rf, alpha=0.3, s=12)
    lim = [min(y_teste.min(), pred_rf.min()), max(y_teste.max(), pred_rf.max())]
    plt.plot(lim, lim, "r--", label="Previsão ideal")
    plt.xlabel("IDEB real (>= 2021)")
    plt.ylabel("IDEB previsto")
    plt.title("Random Forest — real vs. previsto (teste temporal)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "grafico_real_vs_previsto.png", dpi=150)
    plt.close()

    print(f"\nResultados salvos em '{CAMINHO_RESULTADOS.name}' e gráficos exportados.")


if __name__ == "__main__":
    main()