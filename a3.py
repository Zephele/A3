"""Pipeline de modelagem e avaliação do IDEB municipal (A3).

Pergunta: o partido do prefeito ajuda a prever o IDEB do município *além* do
que já é explicado pelo histórico do próprio município (inércia), pela UF e
pela despesa em educação?

Desenho do experimento
----------------------
* Avaliação temporal: treino com anos <= 2019, teste com anos >= 2021.
* O IDEB é medido a cada 2 anos; ``ideb_anterior`` e ``taxa_aprovacao_anterior``
  são os valores do mesmo município e etapa dois anos antes (sem preenchimento
  artificial). A taxa de aprovação do *próprio* ano não é usada: o IDEB é
  construído a partir dela, o que causaria vazamento.
* Baselines: média do treino e persistência (prever o IDEB anterior).
* Dois alvos: ``nivel`` (IDEB) e ``variacao`` (IDEB - IDEB anterior). Em
  ``variacao`` o modelo prevê a mudança e soma ao IDEB anterior; as métricas
  são sempre calculadas na escala do IDEB.
* Ablação em blocos (cada bloco inclui o anterior):
    A0 inércia básica = ideb_anterior + ano + etapa
    A  + taxa de aprovação anterior
    B  + região       = A + UF
    C  + despesa      = B + log(despesa em educação)
    D  + partido      = C + partido do prefeito (siglas harmonizadas)
  A contribuição do partido é a diferença entre D e C, com intervalo de
  confiança por bootstrap em blocos de município.
* Ajuste de hiperparâmetros do Random Forest com validação de janela
  expansiva (respeita a ordem do tempo), nunca com o conjunto de teste. O
  ajuste é feito no bloco D e reaproveitado nos demais; isso favorece D, logo
  a conclusão "o partido não ajuda" é conservadora.

Decisão de pesquisa: harmonização de siglas
-------------------------------------------
Fusões e renomeações de partidos fariam siglas novas (ex.: UNIÃO) aparecerem
só no teste, e o ``OneHotEncoder`` as trataria como "sem partido". O dicionário
``MAPA_PARTIDOS`` leva as siglas antigas para a sucessora. É uma escolha
discutível (ex.: PSL e DEM são organizações diferentes que se fundiram) e deve
ser justificada no relatório; ``HARMONIZAR_PARTIDOS = False`` desliga a regra.

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
DIR_DADOS_BRUTOS = BASE_DIR / "dados" / "brutos"
DIR_DADOS_PROCESSADOS = BASE_DIR / "dados" / "processados"
DIR_RESULTADOS = BASE_DIR / "resultados"
CAMINHO_DADOS = DIR_DADOS_PROCESSADOS / "base_treino_model.csv"
CAMINHO_DESPESA = DIR_DADOS_BRUTOS / "investimento_educacao.csv"
CAMINHO_RESULTADOS = DIR_RESULTADOS / "resultados_modelos.csv"
CAMINHO_PARTIDO = DIR_RESULTADOS / "contribuicao_partido.csv"
CAMINHO_RESUMO_BASE = DIR_RESULTADOS / "resumo_base.csv"

ANO_CORTE_TREINO = 2019
ANO_INICIO_TESTE = 2021
ANO_INICIO_DESPESA = 2013  # a despesa só tem cobertura real a partir daqui
PASSO_IDEB = 2  # o IDEB é divulgado a cada dois anos
COLUNA_ALVO = "ideb"
MODO_PRINCIPAL = "variacao"  # alinhado à pergunta ("evolução" do indicador)
N_BOOTSTRAP = 2000
N_REPETICOES_PERMUTACAO = 20
# Diferenças de MAE menores que isso (em pontos de IDEB) são desprezíveis.
LIMIAR_RELEVANCIA = 0.01

HARMONIZAR_PARTIDOS = True
# Fonte: TSE, "Partidos políticos registrados no TSE" (fusões, incorporações e
# mudanças de nome). As chaves seguem a grafia da base em maiúsculas.
MAPA_PARTIDOS = {
    # Fusões (nasce um partido novo)
    "DEM": "UNIÃO", "PSL": "UNIÃO", "PFL": "UNIÃO",  # DEM+PSL -> UNIÃO (08/02/2022)
    "PTB": "PRD", "PATRIOTA": "PRD",  # PTB+PATRIOTA -> PRD (09/11/2023)
    # Incorporações
    "PRP": "PRD",  # incorporado ao PEN/Patriota (28/03/2019); Patriota -> PRD
    "PEN": "PRD",  # renomeado Patriota (26/04/2018); Patriota -> PRD
    "PPL": "PC DO B",  # 28/05/2019
    "PHS": "PODE",  # 19/09/2019
    "PSC": "PODE",  # 15/06/2023
    "PROS": "SOLIDARIEDADE",  # 14/02/2023
    # Renomeações
    "PTN": "PODE",  # 16/05/2017
    "PT DO B": "AVANTE",  # 12/09/2017
    "PMDB": "MDB",  # 15/05/2018
    "PSDC": "DC",  # 17/05/2018
    "SD": "SOLIDARIEDADE",  # 28/06/2018
    "PR": "PL",  # 09/02/2019
    "PRB": "REPUBLICANOS",  # 15/08/2019
    "PPS": "CIDADANIA",  # 19/09/2019
    "PTC": "AGIR",  # 31/03/2022
    "PMN": "MOBILIZA",  # 05/12/2023
}

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


def adicionar_valores_anteriores(df: pd.DataFrame, chave: list[str]) -> pd.DataFrame:
    """Anexa IDEB, taxa de aprovação e partido de ``PASSO_IDEB`` anos antes.

    Usa junção por ano (e não ``shift``) para nunca puxar uma medição de um
    ano diferente do esperado quando há lacunas no painel.
    """
    anterior = df[chave + ["ano", COLUNA_ALVO, "taxa_aprovacao", "sigla_partido"]].copy()
    anterior["ano"] = anterior["ano"] + PASSO_IDEB
    anterior = anterior.rename(columns={
        COLUNA_ALVO: "ideb_anterior",
        "taxa_aprovacao": "taxa_aprovacao_anterior",
        "sigla_partido": "partido_anterior",
    })
    return df.merge(anterior, on=chave + ["ano"], how="left", validate="one_to_one")


def percentual_partido_inedito(df: pd.DataFrame, coluna: str) -> float:
    """% de linhas do teste cujo partido nunca aparece no treino."""
    treino = df["ano"] <= ANO_CORTE_TREINO
    teste = df["ano"] >= ANO_INICIO_TESTE
    vistos = set(df.loc[treino, coluna].unique())
    return float((~df.loc[teste, coluna].isin(vistos)).mean() * 100)


def percentual_troca_partido(df: pd.DataFrame) -> tuple[float, float]:
    """(% de transições de mandato com troca de partido, % de municípios que trocaram)."""
    mandatos = df.drop_duplicates(["id_municipio", "ano"]).copy()
    mandatos["mandato"] = (mandatos["ano"] - 1) // 4  # 2017-2020 -> 504, etc.
    mandatos = (
        mandatos.sort_values(["id_municipio", "ano"])
        .groupby(["id_municipio", "mandato"], as_index=False)["sigla_partido"]
        .first()
    )
    mandatos["anterior"] = mandatos.groupby("id_municipio")["sigla_partido"].shift(1)
    mandatos["mandato_anterior"] = mandatos.groupby("id_municipio")["mandato"].shift(1)
    consecutivos = mandatos[mandatos["mandato"] - mandatos["mandato_anterior"] == 1]
    trocou = consecutivos["sigla_partido"] != consecutivos["anterior"]
    por_municipio = trocou.groupby(consecutivos["id_municipio"]).any()
    return float(trocou.mean() * 100), float(por_municipio.mean() * 100)


def preparar_dados(caminho: Path, caminho_despesa: Path) -> pd.DataFrame:
    """Limpa, harmoniza siglas, cria valores anteriores e restringe a 2013+."""
    df = carregar_e_cruzar_dados(caminho, caminho_despesa)
    linhas_brutas = len(df)

    colunas_cat = ["sigla_partido", "sigla_uf"]
    colunas_etapa = [c for c in ("ensino", "anos_escolares") if c in df.columns]
    if "anos_escolares" not in colunas_etapa:
        raise ValueError(
            "Coluna 'anos_escolares' ausente: execute 'python preprocess.py' antes."
        )

    for coluna in colunas_cat + colunas_etapa:
        df[coluna] = df[coluna].astype("string").str.strip().str.upper()
        df[coluna] = df[coluna].replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})

    df["sigla_partido_original"] = df["sigla_partido"]
    if HARMONIZAR_PARTIDOS:
        df["sigla_partido"] = df["sigla_partido"].replace(MAPA_PARTIDOS)

    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["despesa_educacao"] = pd.to_numeric(df["despesa_educacao"], errors="coerce")
    df[COLUNA_ALVO] = pd.to_numeric(df[COLUNA_ALVO], errors="coerce")
    df["taxa_aprovacao"] = pd.to_numeric(df["taxa_aprovacao"], errors="coerce")
    df = df.dropna(
        subset=["id_municipio", "ano", COLUNA_ALVO, "taxa_aprovacao"]
        + colunas_cat + colunas_etapa
    )

    chave = ["id_municipio"] + colunas_etapa
    df = df.drop_duplicates(subset=chave + ["ano"]).reset_index(drop=True)
    df = adicionar_valores_anteriores(df, chave)  # antes de cortar o período
    df["variacao_ideb"] = df[COLUNA_ALVO] - df["ideb_anterior"]

    df = df[df["ano"] >= ANO_INICIO_DESPESA].dropna(subset=["despesa_educacao"])
    df = df.dropna(subset=["ideb_anterior", "taxa_aprovacao_anterior"]).copy()

    # Limite superior da despesa estimado só no treino (evita vazamento) e
    # aplicado por truncamento, sem descartar linhas do teste.
    treino = df["ano"] <= ANO_CORTE_TREINO
    limite = df.loc[treino, "despesa_educacao"].quantile(0.995)
    df["log_despesa_educacao"] = np.log1p(df["despesa_educacao"].clip(upper=limite))

    # Colunas constantes não informam nada.
    constantes = [c for c in colunas_etapa if df[c].nunique() < 2]
    if constantes:
        print(f"[INFO] Colunas constantes ignoradas: {constantes}")
    df.attrs["colunas_etapa"] = [c for c in colunas_etapa if c not in constantes]
    df.attrs["linhas_brutas"] = linhas_brutas

    imprimir_resumo_base(df)
    return df


def imprimir_resumo_base(df: pd.DataFrame) -> None:
    """Estatísticas descritivas e porcentagens da base e do experimento."""
    linhas_brutas = df.attrs["linhas_brutas"]
    treino = df["ano"] <= ANO_CORTE_TREINO
    teste = df["ano"] >= ANO_INICIO_TESTE
    pct_inedito_antes = percentual_partido_inedito(df, "sigla_partido_original")
    pct_inedito_depois = percentual_partido_inedito(df, "sigla_partido")
    pct_troca_mandato, pct_mun_trocaram = percentual_troca_partido(df)
    etapas = (df["anos_escolares"].value_counts(normalize=True) * 100).round(2)

    print("=== ANÁLISE EXPLORATÓRIA ===")
    print(f"Municípios: {df['id_municipio'].nunique()} | partidos (após "
          f"harmonização): {df['sigla_partido'].nunique()} | "
          f"UFs: {df['sigla_uf'].nunique()}")
    print(f"Anos: {sorted(df['ano'].unique().tolist())}")
    print(df[["ideb", "ideb_anterior", "variacao_ideb", "taxa_aprovacao_anterior",
              "despesa_educacao"]].describe().round(3).to_string())
    print("\nVariação média do IDEB por ano:")
    print(df.groupby("ano")["variacao_ideb"].agg(["mean", "std", "count"])
          .round(3).to_string())

    print("\n=== PORCENTAGENS DA BASE E DO EXPERIMENTO ===")
    print(f"Retenção após limpeza, 2013+, lag e despesa: "
          f"{len(df) / linhas_brutas * 100:.1f}% ({len(df)}/{linhas_brutas} linhas)")
    print(f"Observações -> treino: {treino.mean() * 100:.1f}% | "
          f"teste: {teste.mean() * 100:.1f}%")
    n_mun = df["id_municipio"].nunique()
    mun_treino = df.loc[treino, "id_municipio"].nunique()
    mun_teste = df.loc[teste, "id_municipio"].nunique()
    print(f"Municípios únicos -> treino: {mun_treino / n_mun * 100:.1f}% | "
          f"teste: {mun_teste / n_mun * 100:.1f}%")
    print(f"Linhas do teste com partido inédito no treino: "
          f"{pct_inedito_antes:.2f}% (siglas originais) -> "
          f"{pct_inedito_depois:.2f}% (harmonizadas)")
    print(f"Troca de partido entre mandatos consecutivos: {pct_troca_mandato:.1f}% "
          f"das transições | {pct_mun_trocaram:.1f}% dos municípios trocaram ao "
          f"menos uma vez")
    print("Composição por etapa (%):")
    print(etapas.to_string())

    resumo = {
        "retencao_pct": len(df) / linhas_brutas * 100,
        "linhas_final": len(df),
        "linhas_brutas": linhas_brutas,
        "treino_pct": treino.mean() * 100,
        "teste_pct": teste.mean() * 100,
        "municipios_treino_pct": mun_treino / n_mun * 100,
        "municipios_teste_pct": mun_teste / n_mun * 100,
        "partido_inedito_original_pct": pct_inedito_antes,
        "partido_inedito_harmonizado_pct": pct_inedito_depois,
        "troca_partido_transicoes_pct": pct_troca_mandato,
        "municipios_com_troca_pct": pct_mun_trocaram,
        **{f"etapa_{k}_pct": float(v) for k, v in etapas.items()},
    }
    pd.Series(resumo, name="valor").round(3).to_csv(
        CAMINHO_RESUMO_BASE, index_label="indicador"
    )


# --------------------------------------------------------------------------
# Modelagem
# --------------------------------------------------------------------------
def definir_blocos(colunas_etapa: list[str]) -> dict[str, dict[str, list[str]]]:
    """Blocos cumulativos da ablação: cada um lista colunas numéricas/categóricas."""
    a0 = {"num": ["ideb_anterior", "ano"], "cat": list(colunas_etapa)}
    a = {"num": a0["num"] + ["taxa_aprovacao_anterior"], "cat": a0["cat"]}
    b = {"num": a["num"], "cat": a["cat"] + ["sigla_uf"]}
    c = {"num": b["num"] + ["log_despesa_educacao"], "cat": b["cat"]}
    d = {"num": c["num"], "cat": c["cat"] + ["sigla_partido"]}
    return {
        "A0 inércia básica": a0,
        "A + taxa aprov. anterior": a,
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
    y = np.asarray(y_real, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "mape_pct": float(np.mean(np.abs(y - y_pred) / y) * 100),  # IDEB >= 1,2
        "r2": float(r2_score(y, y_pred)),
        "r2_pct": float(r2_score(y, y_pred) * 100),
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


def taxa_vitoria_por_municipio(
    dif_erro: np.ndarray, municipios: np.ndarray, tolerancia: float = 1e-9
) -> tuple[float, float, float]:
    """% de municípios em que D erra menos / empata / erra mais que C.

    ``dif_erro`` = erro absoluto de C - erro absoluto de D; o erro de cada
    município é a média dos seus erros no teste.
    """
    por_municipio = pd.Series(dif_erro).groupby(municipios).mean()
    vitorias = (por_municipio > tolerancia).mean() * 100
    empates = (por_municipio.abs() <= tolerancia).mean() * 100
    return float(vitorias), float(empates), float(100 - vitorias - empates)


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------
def main() -> None:
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    df = preparar_dados(CAMINHO_DADOS, CAMINHO_DESPESA)
    blocos = definir_blocos(df.attrs["colunas_etapa"])
    nomes_blocos = list(blocos)

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
    mae_persistencia = resultados[-1]["mae"]

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
    tabela["reducao_mae_vs_persistencia_pct"] = (
        (mae_persistencia - tabela["mae"]) / mae_persistencia * 100
    )
    tabela.to_csv(CAMINHO_RESULTADOS, index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print("\n=== RESULTADOS NO TESTE TEMPORAL (escala do IDEB) ===")
        print(tabela.drop(columns=["r2_pct"]).to_string(index=False))

    # Variância explicada (%) e ganho em pontos percentuais ao longo da ablação.
    print("\n=== R² (%) E GANHO EM PONTOS PERCENTUAIS ENTRE BLOCOS ===")
    for rotulo in ("Random Forest", "Regressão linear"):
        for modo in ("nivel", "variacao"):
            r2s = [tabela[(tabela["modelo"] == rotulo) & (tabela["alvo"] == modo)
                          & (tabela["blocos"] == b)]["r2_pct"].iloc[0]
                   for b in nomes_blocos]
            passos = [f"{nomes_blocos[i]}: {r2s[i] - r2s[i - 1]:+.2f} p.p."
                      for i in range(1, len(r2s))]
            print(f"{rotulo:17s} | alvo={modo:8s} | R² de {r2s[0]:.2f}% (A0) a "
                  f"{r2s[-1]:.2f}% (D) | " + " | ".join(passos))

    # Contribuição do partido: bloco D contra bloco C.
    print("\n=== CONTRIBUIÇÃO DO PARTIDO (D vs C; positivo = partido ajuda) ===")
    municipios = teste["id_municipio"].to_numpy()
    linhas_partido = []
    for rotulo in ("Random Forest", "Regressão linear"):
        for modo in ("nivel", "variacao"):
            erro_c = erros_abs[(rotulo, modo, "C + despesa")]
            erro_d = erros_abs[(rotulo, modo, "D + partido")]
            dif = erro_c - erro_d
            media, lo, hi = intervalo_bootstrap(dif, municipios)
            reducao_pct = media / erro_c.mean() * 100
            vit, emp, der = taxa_vitoria_por_municipio(dif, municipios)
            veredito = ("melhora" if lo > 0 else "piora" if hi < 0
                        else "sem efeito distinguível de zero")
            if max(abs(lo), abs(hi)) < LIMIAR_RELEVANCIA:
                veredito += f" (efeito desprezível: < {LIMIAR_RELEVANCIA} ponto de IDEB)"
            print(f"{rotulo:17s} | alvo={modo:8s} | ganho de MAE = {media:+.4f} "
                  f"(IC95% {lo:+.4f} a {hi:+.4f}) = {reducao_pct:+.2f}% do MAE de C")
            print(f"{'':17s} | municípios: D melhor em {vit:.1f}%, empate "
                  f"{emp:.1f}%, D pior em {der:.1f}% -> {veredito}")
            linhas_partido.append({
                "modelo": rotulo, "alvo": modo, "ganho_mae": media, "ic95_inf": lo,
                "ic95_sup": hi, "reducao_mae_pct": reducao_pct,
                "municipios_d_melhor_pct": vit, "municipios_empate_pct": emp,
                "municipios_d_pior_pct": der, "veredito": veredito,
            })
    pd.DataFrame(linhas_partido).round(5).to_csv(CAMINHO_PARTIDO, index=False)

    # Importância por permutação (variáveis originais, não dummies) no teste.
    print(f"\n=== IMPORTÂNCIA POR PERMUTAÇÃO — RF, alvo={MODO_PRINCIPAL}, bloco D ===")
    pipe_rf = modelos_ajustados[("Random Forest", MODO_PRINCIPAL, "D + partido")]
    colunas_d = bloco_completo["num"] + bloco_completo["cat"]
    perm = permutation_importance(
        pipe_rf, teste[colunas_d], teste[coluna_alvo(MODO_PRINCIPAL)],
        scoring="neg_mean_absolute_error", n_repeats=N_REPETICOES_PERMUTACAO,
        random_state=SEED, n_jobs=-1,
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
    plt.figure(figsize=(9, 4.5))
    plt.barh(comp["modelo"], comp["mae"])
    plt.xlabel("MAE no teste (menor é melhor)")
    plt.title(f"Modelos completos (alvo: {MODO_PRINCIPAL}) vs baselines")
    plt.tight_layout()
    plt.savefig(DIR_RESULTADOS / "grafico_comparacao_modelos.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.barh(imp["variavel"][::-1], imp["aumento_mae"][::-1],
             xerr=imp["desvio"][::-1])
    plt.xlabel("Aumento do MAE ao embaralhar a variável")
    plt.title("Importância por permutação (teste temporal)")
    plt.tight_layout()
    plt.savefig(DIR_RESULTADOS / "grafico_feature_importance.png", dpi=150)
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
    plt.savefig(DIR_RESULTADOS / "grafico_real_vs_previsto.png", dpi=150)
    plt.close()

    print(f"\nSalvos: '{CAMINHO_RESULTADOS.name}', '{CAMINHO_PARTIDO.name}', "
          f"'{CAMINHO_RESUMO_BASE.name}' e os gráficos.")


if __name__ == "__main__":
    main()