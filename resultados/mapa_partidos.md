# Harmonização de siglas partidárias (MAPA_PARTIDOS)

**Fonte:** Tribunal Superior Eleitoral, "Partidos políticos registrados no TSE",
seção de fusões, incorporações e mudanças de nome
(https://www.tse.jus.br/partidos/partidos-registrados-no-tse). Acesso em 14/11/2026.

**Critério:** cada sigla é levada à legenda sucessora registrada no TSE, para que
partidos criados ou renomeados depois de 2019 não apareçam como categoria
desconhecida no conjunto de teste (>= 2021). Com o mapa, as linhas do teste com
partido inédito caem de 4,35% para 0,13%.

**Decisão discutível:** nas fusões (DEM + PSL, PTB + Patriota) nasce juridicamente um
partido novo, e as organizações unidas são historicamente distintas. Por isso a
análise de sensibilidade compara os resultados com e sem harmonização.

| Sigla na base | Sigla harmonizada | Natureza | Data da decisão do TSE |
|---|---|---|---|
| DEM, PSL | UNIÃO | Fusão | 08/02/2022 |
| PTB, PATRIOTA | PRD | Fusão | 09/11/2023 |
| PRP | PRD | Incorporação ao PEN/Patriota; depois fusão | 28/03/2019; 09/11/2023 |
| PEN | PRD | Renomeação para Patriota; depois fusão | 26/04/2018; 09/11/2023 |
| PPL | PC DO B | Incorporação | 28/05/2019 |
| PHS | PODE | Incorporação | 19/09/2019 |
| PSC | PODE | Incorporação | 15/06/2023 |
| PROS | SOLIDARIEDADE | Incorporação | 14/02/2023 |
| PTN | PODE | Renomeação | 16/05/2017 |
| PT DO B | AVANTE | Renomeação | 12/09/2017 |
| PMDB | MDB | Renomeação | 15/05/2018 |
| PSDC | DC | Renomeação | 17/05/2018 |
| SD | SOLIDARIEDADE | Renomeação | 28/06/2018 |
| PR | PL | Renomeação | 09/02/2019 |
| PRB | REPUBLICANOS | Renomeação | 15/08/2019 |
| PPS | CIDADANIA | Renomeação | 19/09/2019 |
| PTC | AGIR | Renomeação | 31/03/2022 |
| PMN | MOBILIZA | Renomeação | 05/12/2023 |
| PFL | UNIÃO | Renomeação para DEM (12/06/2007); depois fusão | 12/06/2007; 08/02/2022 |

O PP mantém a sigla após mudar o nome para Progressistas (11/09/2018), então não
entra no mapa. Em 02/12/2025 o PMB passou a se chamar Democrata; as bases usadas
aqui ainda trazem PMB, e o mapa não o altera.
