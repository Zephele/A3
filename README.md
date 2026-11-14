# Predição do IDEB Municipal e papel do partido do prefeito

Projeto A3 (Sistemas de Controle e Inteligência Artificial).

**Pergunta:** o partido do prefeito ajuda a prever o IDEB municipal *além* do
histórico do próprio município (IDEB anterior), da UF e da despesa em educação?

## Fontes de dados

| Arquivo | Origem | Conteúdo |
|---|---|---|
| `dados/brutos/br_inep_ideb_municipio.csv` | INEP (via Base dos Dados) | IDEB por município, rede e etapa |
| `dados/brutos/bq-results.csv` | TSE (via Base dos Dados) | resultados de eleições municipais |
| `dados/brutos/investimento_educacao.csv` | SICONFI/STN | despesa municipal em educação (2013+) |

> Preencher: links exatos e licença de cada base (necessário para a seção de ética).

## Como reproduzir

```bash
pip install -r requirements.txt
python preprocess.py   # cruza TSE x INEP e gera dados/processados/*.csv
python a3.py           # modela, avalia e gera tabelas e gráficos em resultados/
```

As bases intermediárias ficam em `dados/processados/`. As saídas do `a3.py`
ficam em `resultados/`: `resultados_modelos.csv`, `contribuicao_partido.csv`,
`resumo_base.csv` e os gráficos.

## Metodologia (resumo)

- **Avaliação temporal:** treino com anos <= 2019; teste com anos >= 2021.
- **Valores anteriores:** IDEB e taxa de aprovação do mesmo município e etapa
  dois anos antes (a taxa do próprio ano não entra: vazamento).
- **Harmonização de siglas:** fusões e renomeações levadas à sigla sucessora
  (ex.: DEM e PSL -> UNIÃO), o que reduz de 4,35% para 0,26% as linhas do teste
  com partido inédito. É uma decisão de pesquisa (`MAPA_PARTIDOS` no código).
- **Baselines:** média do treino e persistência (IDEB anterior).
- **Alvos:** nível do IDEB e variação (IDEB - IDEB anterior).
- **Ablação em blocos:** A0 inércia -> A + taxa de aprovação anterior ->
  B + UF -> C + despesa -> D + partido. O efeito do partido é D vs C, com IC 95%
  por bootstrap em blocos de município e taxa de vitória por município.
- **Hiperparâmetros:** busca aleatória com janelas expansivas (passado -> futuro),
  feita no bloco D e reaproveitada nos demais (favorece D).
- **Split aleatório 80/20:** apenas diagnóstico de vazamento (R² 0,84).

## Resultados (teste temporal, escala do IDEB)

| Modelo | MAE | MAPE | R² |
|---|---|---|---|
| Baseline média | 0,929 | 16,2% | -0,31 |
| Baseline persistência | 0,450 | 8,3% | 0,665 |
| Random Forest, só inércia (A0) | 0,407 | 7,6% | 0,723 |
| Random Forest completo, com partido (D) | 0,406 | 7,6% | 0,727 |

- Os modelos reduzem em ~10% o erro da persistência.
- O partido não melhora a previsão: ganho de MAE de D sobre C de +0,0001
  (IC 95% -0,0001 a +0,0002); D vence em 50,5% dos municípios (RF, alvo variação).
- A taxa de aprovação anterior também não acrescenta (o IDEB anterior já a resume).
- A UF é o bloco que mais acrescenta (cerca de +0,6 p.p. de R²).

## Base e experimento (porcentagens)

- Retenção após limpeza, 2013+, lag e despesa: 60,0% (53.183 de 88.630 linhas).
- Observações: 57,9% treino, 42,1% teste.
- Troca de partido entre mandatos consecutivos: 69,6% das transições; 93,5% dos
  municípios trocaram ao menos uma vez.
- Etapas: 63,2% anos iniciais, 36,6% finais, 0,2% "todos (1-4)".

## Limitações

- Partido do prefeito eleito; trocas de partido e mandatos-tampão não modelados.
- Harmonização de siglas é uma escolha discutível e deve ser justificada.
- Despesa nominal (não deflacionada) e absoluta (não per capita/aluno).
- A pandemia (SAEB 2021) cai inteira no período de teste.
- Associação preditiva não é causalidade: partido, região e riqueza do
  município estão correlacionados.