"""Análise de sensibilidade do efeito do partido (e de outras variáveis candidatas).

Reaproveita os dados, os blocos e as funções do ``a3.py``, de modo que os números
são diretamente comparáveis aos do pipeline principal. Todas as variantes usam o
mesmo desenho: Random Forest prevendo a variação do IDEB, treino <= 2019, teste
>= 2021, hiperparâmetros ajustados uma vez (janela expansiva) no bloco D do
``a3.py`` e reaproveitados aqui.

Parte 1 - Efeito do PARTIDO em cada especificação (bloco D vs bloco C da MESMA
especificação, para isolar o partido):
    1. siglas originais (sem harmonização)
    2. siglas harmonizadas (MAPA_PARTIDOS)
    3. harmonizadas, com ``ano_mandato`` nos dois blocos
    4. harmonizadas, com ``ano_mandato`` nos dois blocos e ``trocou_partido`` no D

Parte 2 - Efeito de outras variáveis sobre um modelo SEM partido:
    5. ``ano_mandato`` sobre o bloco C
    6. ``taxa_aprovacao_anterior`` sobre o bloco A0 (inércia básica)

Em todas, ganho = MAE(sem a variável) - MAE(com a variável); positivo = ajuda.
O IC 95% é por bootstrap em blocos de município e o veredito é calculado a partir
do IC (nada é escrito à mão).

Uso:  python analise_sensibilidade.py
Saída: tabela_sensibilidade.csv e tabela_sensibilidade.md
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import a3

BASE_DIR = Path(__file__).resolve().parent
DIR_RESULTADOS = BASE_DIR / "resultados"
SAIDA_CSV = DIR_RESULTADOS / "tabela_sensibilidade.csv"
SAIDA_MD = DIR_RESULTADOS / "tabela_sensibilidade.md"
MODO = "variacao"


def carregar_dados() -> pd.DataFrame:
    """Dados do a3.py (silencioso) mais as variáveis derivadas desta análise."""
    with contextlib.redirect_stdout(io.StringIO()):
        df = a3.preparar_dados(a3.CAMINHO_DADOS, a3.CAMINHO_DESPESA)
    # Ciclo eleitoral municipal: IDEB de 2013, 2017, 2021, 2025 cai no 1º ano de
    # mandato; 2015, 2019, 2023 no 3º ano.
    df["ano_mandato"] = ((df["ano"] - 1) % 4 + 1).astype(int).astype(str)
    df["trocou_partido"] = (df["sigla_partido"] != df["partido_anterior"]).astype(int)
    return df


def bloco(base: dict, num: list[str] = (), cat: list[str] = ()) -> dict:
    """Bloco derivado: acrescenta colunas numéricas e/ou categóricas."""
    return {"num": base["num"] + list(num), "cat": base["cat"] + list(cat)}


def erros_do_bloco(chave, cache, treino, teste, params, blk) -> np.ndarray:
    """Erro absoluto (na escala do IDEB) de um RF treinado no bloco dado."""
    if chave not in cache:
        colunas = blk["num"] + blk["cat"]
        pipe = a3.criar_pipeline("rf", blk, **params)
        pipe.fit(treino[colunas], treino[a3.coluna_alvo(MODO)])
        pred = a3.para_escala_ideb(pipe.predict(teste[colunas]), teste, MODO)
        cache[chave] = (np.abs(teste[a3.COLUNA_ALVO].to_numpy() - pred), pred)
    return cache[chave][0]


def classificar(lo: float, hi: float) -> str:
    """Veredito calculado a partir do IC 95% do ganho de MAE."""
    texto = ("ajuda" if lo > 0 else "piora" if hi < 0
             else "sem efeito distinguível de zero")
    if max(abs(lo), abs(hi)) < a3.LIMIAR_RELEVANCIA:
        texto += f" (desprezível: < {a3.LIMIAR_RELEVANCIA} ponto de IDEB)"
    return texto


def comparar(analise, comparacao, chave_sem, blk_sem, chave_com, blk_com,
             contexto) -> dict:
    cache, treino, teste, params = contexto
    erro_sem = erros_do_bloco(chave_sem, cache, treino, teste, params, blk_sem)
    erro_com = erros_do_bloco(chave_com, cache, treino, teste, params, blk_com)
    ganho, lo, hi = a3.intervalo_bootstrap(
        erro_sem - erro_com, teste["id_municipio"].to_numpy()
    )
    y = teste[a3.COLUNA_ALVO].to_numpy()
    r2_sem = a3.metricas(y, cache[chave_sem][1])["r2"]
    r2_com = a3.metricas(y, cache[chave_com][1])["r2"]
    return {
        "analise": analise,
        "comparacao": comparacao,
        "mae_sem": erro_sem.mean(),
        "mae_com": erro_com.mean(),
        "ganho_mae": ganho,
        "ic95_inf": lo,
        "ic95_sup": hi,
        "ganho_pct_do_mae": ganho / erro_sem.mean() * 100,
        "delta_r2_pp": (r2_com - r2_sem) * 100,
        "veredito": classificar(lo, hi),
    }


def formatar_md(tabela: pd.DataFrame) -> str:
    """Markdown com vírgula decimal, sem depender de ``tabulate``."""
    def n(x, casas=4, sinal=True):
        t = f"{x:+.{casas}f}" if sinal else f"{x:.{casas}f}"
        return t.replace(".", ",")

    linhas = [
        "| Análise | Comparação | MAE sem | MAE com | Ganho de MAE | IC 95% | "
        "Ganho (% do MAE) | ΔR² (p.p.) | Veredito |",
        "| :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | :--- |",
    ]
    for r in tabela.itertuples():
        linhas.append(
            f"| {r.analise} | {r.comparacao} | {n(r.mae_sem, sinal=False)} | "
            f"{n(r.mae_com, sinal=False)} | {n(r.ganho_mae)} | "
            f"{n(r.ic95_inf)} a {n(r.ic95_sup)} | {n(r.ganho_pct_do_mae, 2)}% | "
            f"{n(r.delta_r2_pp, 2)} | {r.veredito} |"
        )
    return "\n".join(linhas) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    df = carregar_dados()
    blocos = a3.definir_blocos(df.attrs["colunas_etapa"])
    treino = df[df["ano"] <= a3.ANO_CORTE_TREINO].reset_index(drop=True)
    teste = df[df["ano"] >= a3.ANO_INICIO_TESTE].reset_index(drop=True)

    print("Ajustando hiperparâmetros no bloco D do a3.py (janela expansiva)...")
    with contextlib.redirect_stdout(io.StringIO()):
        params = a3.ajustar_hiperparametros(treino, blocos["D + partido"], MODO)
    contexto = ({}, treino, teste, params)

    c = blocos["C + despesa"]
    c_mandato = bloco(c, cat=["ano_mandato"])
    resultados = [
        # Parte 1: efeito do partido, D vs C da mesma especificação.
        comparar("Partido", "1. Siglas originais",
                 "C", c, "C+orig", bloco(c, cat=["sigla_partido_original"]), contexto),
        comparar("Partido", "2. Siglas harmonizadas",
                 "C", c, "C+harm", bloco(c, cat=["sigla_partido"]), contexto),
        comparar("Partido", "3. Harmonizadas, com ano_mandato nos dois blocos",
                 "C+mand", c_mandato, "C+mand+harm",
                 bloco(c_mandato, cat=["sigla_partido"]), contexto),
        comparar("Partido", "4. Harmonizadas, ano_mandato + trocou_partido no D",
                 "C+mand", c_mandato, "C+mand+harm+troca",
                 bloco(c_mandato, num=["trocou_partido"], cat=["sigla_partido"]),
                 contexto),
        # Parte 2: outras variáveis, sobre modelos sem partido.
        comparar("Outra variável", "5. ano_mandato sobre o bloco C",
                 "C", c, "C+mand", c_mandato, contexto),
        comparar("Outra variável", "6. taxa_aprovacao_anterior sobre o bloco A0",
                 "A0", blocos["A0 inércia básica"], "A", blocos["A + taxa aprov. anterior"],
                 contexto),
    ]

    tabela = pd.DataFrame(resultados)
    tabela.round(6).to_csv(SAIDA_CSV, index=False)
    md = formatar_md(tabela)
    SAIDA_MD.write_text(md, encoding="utf-8")
    print("\n=== ANÁLISE DE SENSIBILIDADE (ganho = MAE sem - MAE com; positivo ajuda) ===\n")
    print(md)
    print(f"Salvos: '{SAIDA_CSV.name}' e '{SAIDA_MD.name}'.")


if __name__ == "__main__":
    main()
