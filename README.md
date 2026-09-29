# AgroSearch
Autor: Fernando Amorim Pontes Neto. Trabalho individual.

## Execução
Requer Python 3.12 e acesso à internet apenas para instalar as dependências.
Na pasta deste projeto:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run agrosearch_app.py
```

No Windows, ative com `.venv\Scripts\activate`.

## Uso e implementação
O corpus de cinco documentos está no próprio script. Digite uma consulta,
experimente os controles de stopwords e stemming e inspecione o pipeline,
o índice invertido e a memória dos cálculos. A opção Cosseno implementa o bônus.
TF é frequência relativa; IDF é ln((N+1)/(df+1))+1. O score principal soma os
pesos dos termos únicos da consulta encontrados em cada documento.
A ordenação por cosseno usa o vetor TF-IDF da consulta, no vocabulário do corpus.
Consultas vazias, sem tokens ou com termos desconhecidos não indicam vencedor.
Empates são informados e mantêm ordem por ID.

TF-IDF, índice e cosseno usam apenas a biblioteca padrão. NLTK fornece somente
o stemmer português; não se usa scikit-learn nem TfidfVectorizer neste projeto.
Não é preciso executar nltk.download. A lista de stopwords é explícita no código.

## Entrega
O arquivo agrosearch_app.py é a aplicação única solicitada. RELATORIO.pdf contém
o relatório de até duas páginas. requirements.txt e este guia apoiam a reprodução.

## Estado da entrega
Aplicação implementada, testes matemáticos aprovados e relatório de duas páginas revisado. Os testes de interface das funções principais (consulta, controles de stopwords/stemming, ordenação por cosseno e entradas vazias) passaram na execução anterior.

## Testes
Execute `python testes.py`. Os testes de lógica não baixam modelos.

## GitHub
Este diretório está versionado com código, relatório e requirements. Ambientes virtuais e pesos de modelos não são versionados (ver `.gitignore`).
