# Predição do IDEB Municipal e papel do partido do prefeito

Projeto A3 (Sistemas de Controle e Inteligência Artificial).

**Pergunta:** o partido do prefeito ajuda a prever o IDEB municipal *além* do
histórico do próprio município (IDEB anterior), da UF e da despesa em educação?

## Fontes de dados

| Arquivo | Origem | Conteúdo |
|---|---|---|
| `br_inep_ideb_municipio.csv` | INEP (via Base dos Dados) | IDEB por município, rede e etapa |
| `bq-results.csv` | TSE (via Base dos Dados) | resultados de eleições municipais |
| `investimento_educacao.csv` | SICONFI/STN | despesa municipal em educação (2013+) |

> Preencher: links exatos e licença de cada base (necessário para a seção de ética).

## Como reproduzir

```bash
pip install -r requirements.txt
python preprocess.py   # cruza TSE x INEP e gera base_treino_model.csv
python a3.py           # modela, avalia e gera resultados_modelos.csv + gráficos
```

## Metodologia (resumo)

- **Avaliação temporal:** treino com anos <= 2019; teste com anos >= 2021.
- **IDEB anterior:** IDEB do mesmo município e etapa dois anos antes.
- **Baselines:** média do treino e persistência (IDEB anterior).
- **Alvos:** nível do IDEB e variação (IDEB - IDEB anterior).
- **Ablação em blocos:** A inércia -> B + UF -> C + despesa -> D + partido.
  O efeito do partido é a diferença D vs C, com IC 95% por bootstrap em
  blocos de município.
- **Ajuste de hiperparâmetros:** busca aleatória com janelas expansivas
  (treina no passado, valida no futuro).
- **Split aleatório 80/20:** apenas diagnóstico de vazamento.

## Resultados (teste temporal, escala do IDEB)

| Modelo | MAE | R² |
|---|---|---|
| Baseline média | 0,929 | -0,31 |
| Baseline persistência | 0,450 | 0,665 |
| Random Forest (A, só inércia + etapa) | 0,407 | 0,719 |
| Random Forest (D, com partido) | 0,407 | 0,726 |

O partido não melhora a previsão: o ganho de MAE do bloco D sobre o C é
desprezível (< 0,01 ponto de IDEB). Detalhes em `resultados_modelos.csv`.

## Limitações

- Partido do prefeito eleito; trocas de partido e mandatos-tampão não modelados.
- Despesa nominal (não deflacionada) e absoluta (não per capita/aluno).
- A pandemia (SAEB 2021) cai inteira no período de teste.
- Associação preditiva não é causalidade: partido, região e riqueza do
  município estão correlacionados.