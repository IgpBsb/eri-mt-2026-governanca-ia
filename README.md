# Governança Digital e Adoção de IA no Judiciário Brasileiro

Código e dados derivados do artigo *Governança Digital e Adoção de Inteligência Artificial no Judiciário Brasileiro: uma Análise Baseada na Integração de Dados Públicos* (submetido à XV ERI-MT 2026).

> Identificação da autoria omitida durante a avaliação às cegas; será incluída após a aprovação e a publicação.

## O que o pipeline faz
`analise/pipeline.py` integra duas bases públicas do Conselho Nacional de Justiça (CNJ):
- **iGovTIC-JUD 2021–2026**: apenas as 28 perguntas da Área de TIC;
- **Pesquisa/Painel de Projetos de IA no Poder Judiciário 2022–2024**.

Etapas: ingestão, *record linkage* de siglas, harmonização de itens entre ciclos, escores de governança por objetivo COBIT 2019 (EDM01, APO02, APO07, APO12, APO13), painel órgão-ano, testes (Mann-Whitney, logit com controle de segmento, kappa de Cohen), teste de direção reversa e modelo preditivo exploratório (validação cruzada 5×20).

## Estrutura
```
analise/pipeline.py              # pipeline completo
analise/output/painel_orgao_ano.csv   # painel integrado (dado derivado)
analise/output/qp*.csv, modelo_preditivo.csv, resultados.json
analise/output/fig_*.pdf|png     # figuras do artigo
```

## Como reproduzir
1. Python 3.11+ e `pip install -r requirements.txt`.
2. Baixe as bases públicas do CNJ (resultados do iGovTIC-JUD 2021–2026 e bases da Pesquisa de IA 2022–2024) e organize-as como:
```
Base Dados/iGovTIC-JUD/iGovTIC-JUD {ano} - Resultado Geral - Limpo.xlsx
Base Dados/Painel de Projetos de IA no Poder Judiciário/{2022-Projetos-IA.csv, 2023-Respostas-Pesquisa-IA.xlsx, 2024-Projetos-IA.xlsx}
```
3. Rode `python -I analise/pipeline.py <raiz>`, em que `<raiz>` contém a pasta `Base Dados`.

As bases brutas não são redistribuídas aqui; o painel derivado em `analise/output/` permite conferir todos os números do artigo.
