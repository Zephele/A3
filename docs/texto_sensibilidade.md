### Análise de sensibilidade

Para verificar se a ausência de ganho preditivo do partido do prefeito decorre de
escolhas de modelagem, executamos o script `analise_sensibilidade.py`. Todas as
variantes usam o mesmo desenho do modelo principal (Random Forest prevendo a
variação do IDEB; treino até 2019, teste a partir de 2021). Em cada linha, o ganho
é o MAE do modelo sem a variável menos o MAE do modelo com ela, e o intervalo de
confiança de 95% vem de *bootstrap* em blocos de município.

| Análise | Comparação | MAE sem | MAE com | Ganho de MAE | IC 95% | Ganho (% do MAE) | ΔR² (p.p.) | Veredito |
| :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | :--- |
| Partido | 1. Siglas originais | 0,4060 | 0,4060 | +0,0001 | -0,0001 a +0,0003 | +0,02% | +0,01 | sem efeito distinguível de zero (desprezível: < 0.01 ponto de IDEB) |
| Partido | 2. Siglas harmonizadas | 0,4060 | 0,4061 | -0,0001 | -0,0002 a +0,0001 | -0,01% | +0,00 | sem efeito distinguível de zero (desprezível: < 0.01 ponto de IDEB) |
| Partido | 3. Harmonizadas, com ano_mandato nos dois blocos | 0,4216 | 0,4203 | +0,0014 | +0,0012 a +0,0015 | +0,32% | +0,15 | ajuda (desprezível: < 0.01 ponto de IDEB) |
| Partido | 4. Harmonizadas, ano_mandato + trocou_partido no D | 0,4216 | 0,4200 | +0,0016 | +0,0013 a +0,0019 | +0,39% | +0,16 | ajuda (desprezível: < 0.01 ponto de IDEB) |
| Outra variável | 5. ano_mandato sobre o bloco C | 0,4060 | 0,4216 | -0,0156 | -0,0167 a -0,0144 | -3,84% | -1,94 | piora |
| Outra variável | 6. taxa_aprovacao_anterior sobre o bloco A0 | 0,4066 | 0,4083 | -0,0017 | -0,0022 a -0,0011 | -0,42% | -0,13 | piora (desprezível: < 0.01 ponto de IDEB) |

**Resultados.**

1. **Partido, com e sem harmonização de siglas (linhas 1 e 2).** O ganho de MAE é
   de +0,0001 com as siglas originais (IC 95%: -0,0001 a +0,0003) e de -0,0001 com
   as harmonizadas (IC 95%: -0,0002 a +0,0001). Em ambos os casos o intervalo
   inclui zero e a magnitude é inferior a 0,001 ponto de IDEB. A conclusão não
   depende da harmonização das siglas.

2. **Ano do mandato (linha 5).** Incluir o indicador de primeiro ou terceiro ano de
   mandato piora o modelo sem partido: o MAE sobe de 0,4060 para 0,4216 (+3,8%; IC
   95% do ganho: -0,0167 a -0,0144). Os dados sugerem por quê. No treino, a
   variação média do IDEB é maior no 3º ano de mandato (+0,25) do que no 1º (+0,16).
   No teste, o 1º ano de mandato reúne 2021 (variação média de -0,06) e 2025
   (+0,37), com sinais opostos, e o padrão aprendido não se mantém. Uma hipótese é o
   efeito da pandemia em 2021 e a recuperação posterior, mas ela não foi testada.

3. **Partido dentro do modelo com ano do mandato (linhas 3 e 4).** Nessa
   especificação o partido apresenta um ganho pequeno, porém distinguível de zero:
   +0,0014 (IC 95%: +0,0012 a +0,0015), ou 0,3% do MAE, e +0,0016 quando se
   acrescenta a variável `trocou_partido`. Não há contradição com os demais
   resultados: o modelo com ano do mandato é pior que o modelo sem ele (0,4203
   contra 0,4060), e o partido apenas recupera uma pequena parte do erro que essa
   variável introduziu. Em um teste preliminar com outros hiperparâmetros o sinal
   desse ganho foi o oposto, o que reforça que o efeito é desprezível e sensível a
   escolhas de modelagem.

4. **Taxa de aprovação defasada (linha 6).** Acrescentar `taxa_aprovacao_anterior`
   ao bloco de inércia básica não melhora a previsão: o MAE sobe de 0,4066 para
   0,4083 e o R² cai 0,13 ponto percentual. O IDEB anterior já resume a maior
   parte da informação que essa variável poderia trazer.

**Limitações.** Os hiperparâmetros foram ajustados uma vez, no bloco completo, e
reaproveitados nas demais variantes, o que favorece a especificação com partido.
Usamos uma única família de modelos e uma única divisão temporal. Os resultados
medem associação preditiva e não permitem inferir efeito causal do partido sobre o
desempenho educacional.
