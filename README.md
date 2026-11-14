# Predição do IDEB Municipal e papel do partido do prefeito

Projeto A3 (Sistemas de Controle e Inteligência Artificial).

**Pergunta:** o partido do prefeito ajuda a prever o IDEB municipal *além* do
histórico do próprio município (IDEB anterior), da UF e da despesa em educação?

## Fontes de dados

| Base | Fonte original | Links | Licença / termos | Data de acesso | Limitações |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IDEB** (`br_inep_ideb_municipio.csv`) | INEP/MEC, via Base dos Dados | [Base dos Dados](https://basedosdados.org/dataset/96eab476-5d30-459b-82be-f888d4d0d6b9) · [INEP: resultados do IDEB](https://www.gov.br/inep/pt-br/areas-de-atuacao/pesquisas-estatisticas-e-indicadores/ideb/resultados) · [INEP: dados abertos](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos) | Dado público federal (LAI, Lei 12.527/2011; Decreto 8.777/2016): reúso com citação da fonte. Conferir o campo de licença na página da tabela. | 21/10/2026 | Medido a cada 2 anos (anos ímpares); não capta variação anual |
| **Eleições** (`bq-results.csv`) | TSE, via Base dos Dados | [Base dos Dados](https://basedosdados.org/dataset/eef764df-bde8-4905-b115-6fc23b6ba9d6) · [Portal de Dados Abertos do TSE](https://dadosabertos.tse.jus.br/) | Creative Commons Atribuição (CC BY), conforme o Portal de Dados Abertos do TSE | 26/10/2026 | Considera só o partido do prefeito eleito (sem coligações, trocas de partido no mandato ou mandatos-tampão) |
| **Despesa em educação** (`investimento_educacao.csv`) | SICONFI/FINBRA, Tesouro Nacional, via Base dos Dados | [Base dos Dados](https://basedosdados.org/dataset/5a3dec52-8740-460e-b31d-0e0347979da0) · [SICONFI: consulta FINBRA](https://siconfi.tesouro.gov.br/siconfi/pages/public/consulta_finbra/finbra_list.jsf) · [Tesouro Transparente](https://www.tesourotransparente.gov.br/consultas/consultas-siconfi/siconfi-finbra-demonstrativos-de-contas-anuais) | Dado público (LAI; LRF, LC 101/2000). Conferir a licença na página do conjunto; citar a fonte. | 09/11/2026 | Valores nominais (sem deflação) e absolutos (não per capita nem por aluno); cobertura utilizada a partir de 2013; depende da declaração dos municípios |

> A coleta dos CSVs `bq-results.csv` e `investimento_educacao.csv` foi feita
> manualmente na Base dos Dados (BigQuery). Os arquivos brutos estão versionados
> em `dados/brutos/`, portanto a análise pode ser reproduzida a partir dos
> próprios arquivos; os SQLs históricos da coleta não foram preservados.

## Como reproduzir

```bash
pip install -r requirements.txt
python preprocess.py            # gera dados/processados/*.csv
python a3.py                    # gera tabelas e gráficos em resultados/
python analise_sensibilidade.py # testa variantes e salva em resultados/
```

As bases intermediárias ficam em `dados/processados/`. As saídas do `a3.py` e
da análise de sensibilidade ficam em `resultados/`: tabelas CSV, Markdown e
gráficos. O mapa de siglas partidárias, com as datas do TSE, está em
`docs/mapa_partidos.md`.

## Metodologia (resumo)

- **Avaliação temporal:** treino com anos <= 2019; teste com anos >= 2021.
- **Valores anteriores:** IDEB e taxa de aprovação do mesmo município e etapa
  dois anos antes (a taxa do próprio ano não entra: vazamento).
- **Harmonização de siglas:** fusões, incorporações e renomeações levadas à
  sigla sucessora registrada no TSE (ex.: DEM e PSL -> UNIÃO; ver `docs/mapa_partidos.md`), o que reduz de
  4,35% para 0,13% as linhas do teste com partido inédito. É uma decisão de
  pesquisa (`MAPA_PARTIDOS` no código; documentação em `docs/mapa_partidos.md`); os resultados são comparados com e sem
  harmonização.
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
- O partido não melhora a previsão: ganho de MAE de D sobre C de -0,0001
  (IC 95% -0,0002 a +0,0001); D erra menos em 51,3% dos municípios (RF, alvo
  variação), praticamente um empate.
- Importância por permutação (aumento do MAE): IDEB anterior 0,048; etapa 0,011;
  UF 0,004; despesa 0,0015; partido ≈ 0.
- A taxa de aprovação anterior também não acrescenta (o IDEB anterior já a resume).
- A UF é o bloco que mais acrescenta (cerca de +0,6 p.p. de R²).
- Split aleatório 80/20 (apenas diagnóstico de vazamento): R² 0,84, contra 0,73
  no teste temporal.

## Análise de sensibilidade

Resultados completos em `resultados/tabela_sensibilidade.md`; texto de discussão
em `docs/texto_sensibilidade.md`.

- Com e sem harmonização de siglas, o ganho do partido é ≈ 0 (|ganho| < 0,001
  ponto de IDEB, IC incluindo zero).
- Incluir o ano do mandato piora o modelo sem partido (MAE de 0,4060 para 0,4216).
  No teste, o 1º ano de mandato reúne 2021 (variação média -0,06) e 2025 (+0,37),
  e o padrão do treino não se mantém.
- Dentro do modelo com ano do mandato, o partido recupera uma fração mínima do
  erro (+0,0014), e o modelo continua pior que o sem essa variável.
- A taxa de aprovação anterior piora levemente a previsão (MAE +0,0017).

## Base e experimento (porcentagens)

- Retenção após limpeza, 2013+, lag e despesa: 60,0% (53.183 de 88.630 linhas).
- Observações: 57,9% treino, 42,1% teste.
- Troca de partido entre mandatos consecutivos: 69,6% das transições; 93,5% dos
  municípios trocaram ao menos uma vez.
- Etapas: 63,2% anos iniciais, 36,6% finais, 0,2% "todos (1-4)".

## Ética e governança de dados

Todas as bases são públicas e gratuitas, de órgãos oficiais (INEP, TSE e
Secretaria do Tesouro Nacional), obtidas por meio da Base dos Dados. O INEP e o
Tesouro estão sujeitos à Lei de Acesso à Informação (Lei 12.527/2011) e à
Política de Dados Abertos do Poder Executivo Federal (Decreto 8.777/2016); o TSE,
como órgão do Judiciário, tem política própria (Portaria-TSE 93/2021) e publica
seus conjuntos sob CC BY. Atribuímos as fontes originais e a Base dos Dados.

Os dados estão no nível de município, rede e etapa de ensino. O único dado
individual é o partido do prefeito eleito, informação pública de eleições, e
não há microdados pessoais de cidadãos ou dados sensíveis (LGPD, Lei 13.709/2018).

O risco principal é de interpretação: os resultados mostram associação
preditiva, não causalidade. Partido, região e porte do município estão
correlacionados, e um erro de leitura poderia atribuir a um partido o mérito ou
a culpa por resultados educacionais. Erros do modelo também têm impacto se ele
for usado para priorizar recursos; por isso o resultado aqui é analítico, não
um instrumento de decisão.

## Limitações

- Partido do prefeito eleito; trocas de partido e mandatos-tampão não modelados.
- Harmonização de siglas é uma escolha discutível e deve ser justificada.
- Despesa nominal (não deflacionada) e absoluta (não per capita/aluno).
- A pandemia (SAEB 2021) cai inteira no período de teste.
- IDEB bienal: a previsão é sempre de dois anos à frente.
- Associação preditiva não é causalidade.