# Predição do IDEB Municipal

Projeto de Ciência de Dados que avalia o poder preditivo do investimento
público em educação e da filiação partidária sobre o IDEB da rede municipal
no Brasil.

## Fontes de dados

- `br_inep_ideb_municipio.csv`: INEP, com notas por município, rede e etapa.
- `bq-results.csv`: TSE, com resultados eleitorais.
- `investimento_educacao.csv`: SICONFI/STN, com despesa municipal em educação.

## Execução

```bash
pip install -r requirements.txt
python preprocess.py
python a3.py
```

O pré-processamento separa as etapas de ensino e remove duplicidades na chave
município-etapa-ano. O modelo usa apenas despesas com cobertura real (2013+),
logaritmo da despesa, avaliação temporal (treino até 2019 e teste a partir de
2021), baseline de persistência e validação agrupada por município.

O split aleatório 80/20 é mantido somente como diagnóstico de possível
vazamento, não como avaliação principal. As importâncias do modelo indicam
associação preditiva, não causalidade.
