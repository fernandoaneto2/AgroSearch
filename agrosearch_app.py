"""AgroSearch — Fernando Amorim Pontes Neto.
Execute: streamlit run agrosearch_app.py
TF-IDF, índice invertido e cosseno implementados com a biblioteca padrão.
NLTK é usado exclusivamente para o stemmer português (sem downloads de corpora).
"""
from collections import Counter, defaultdict
import math
import re
import unicodedata
from nltk.stem.snowball import PortugueseStemmer

DOCUMENTOS = {
    'Doc 1': 'A soja requer irrigação constante durante o período de floração para garantir a produtividade.',
    'Doc 2': 'O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.',
    'Doc 3': 'A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.',
    'Doc 4': 'Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.',
    'Doc 5': 'A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.',
}
# Lista explícita e auditável; não depende do corpus stopwords do NLTK.
STOPWORDS = set('a as o os um uma uns umas de da das do dos em na nas no nos ao aos e ou que se com por para pelo pela pelos pelas durante ser estar ter foi sao e'.split())
STEMMER = PortugueseStemmer()

def normalizar(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto.lower())
                   if unicodedata.category(c) != 'Mn')

def preprocessar(texto, remover_stopwords=True, stemming=True):
    tokens = re.findall(r'[a-z0-9]+', normalizar(texto))
    if remover_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stemming:
        tokens = [normalizar(STEMMER.stem(t)) for t in tokens]
    return tokens

def construir_indice(documentos, remover_stopwords=True, stemming=True):
    tokens = {i: preprocessar(t, remover_stopwords, stemming) for i, t in documentos.items()}
    indice = defaultdict(list)
    for i, ts in tokens.items():
        for termo in sorted(set(ts)):
            indice[termo].append(i)
    n = len(documentos)
    # IDF suavizada e positiva. TF é frequência relativa no documento.
    idf = {t: math.log((n + 1) / (len(ids) + 1)) + 1 for t, ids in indice.items()}
    vetores = {i: {t: (c / len(ts)) * idf[t] for t, c in Counter(ts).items()}
               for i, ts in tokens.items()}
    return tokens, dict(sorted(indice.items())), idf, vetores

def cosseno(a, b):
    norma = math.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
    return sum(v*b.get(t, 0) for t, v in a.items()) / norma if norma else 0.0

def buscar(consulta, documentos=DOCUMENTOS, remover_stopwords=True, stemming=True):
    tokens, indice, idf, vetores = construir_indice(documentos, remover_stopwords, stemming)
    qt = preprocessar(consulta, remover_stopwords, stemming)
    # OOV não tem IDF aprendida: o vetor da query usa somente o vocabulário do corpus.
    conhecidos = [t for t in qt if t in idf]
    qv = {t: c / len(conhecidos) * idf[t] for t, c in Counter(conhecidos).items()}
    candidatos = {i for t in set(conhecidos) for i in indice[t]}
    linhas = [{'Documento': i, 'TF-IDF acumulado': sum(vetores[i].get(t, 0) for t in set(conhecidos)),
               'Cosseno': cosseno(qv, vetores[i]), 'Texto': texto}
              for i, texto in documentos.items()]
    linhas.sort(key=lambda r: (-r['TF-IDF acumulado'], r['Documento']))
    return linhas, qt, sorted(set(qt) - set(idf)), candidatos

def main():
    import streamlit as st
    import pandas as pd
    st.set_page_config(page_title='AgroSearch', page_icon='🌱', layout='wide')
    st.title('AgroSearch')
    st.caption('Busca em manuais agrícolas • Fernando Amorim Pontes Neto')
    sw = st.sidebar.checkbox('Remover stopwords', value=True)
    stem = st.sidebar.checkbox('Aplicar stemming', value=True)
    modo = st.sidebar.radio('Ordenar resultados por', ['TF-IDF acumulado', 'Cosseno'])
    st.sidebar.caption('Os controles reconstroem o vocabulário e o índice imediatamente.')
    tokens, indice, idf, vetores = construir_indice(DOCUMENTOS, sw, stem)
    c1, c2, c3 = st.columns(3)
    c1.metric('Documentos', len(DOCUMENTOS)); c2.metric('Vocabulário', len(indice))
    c3.metric('Tokens', sum(map(len, tokens.values())))
    busca, pipeline, invertido, calculos = st.tabs(['Busca', 'Pré-processamento', 'Índice invertido', 'Cálculos'])
    with busca:
        consulta = st.text_input('O que você procura?', 'irrigação soja')
        linhas, qt, oov, candidatos = buscar(consulta, DOCUMENTOS, sw, stem)
        linhas.sort(key=lambda r: (-r[modo], r['Documento']))
        st.write('Consulta processada:', qt)
        if not qt:
            st.info('Digite termos válidos; a consulta ficou vazia após o pré-processamento.')
        elif not candidatos:
            st.warning('Nenhum termo foi encontrado no corpus. Não há documento vencedor.')
        else:
            maior = linhas[0][modo]
            vencedores = [r['Documento'] for r in linhas if math.isclose(r[modo], maior, rel_tol=1e-12)]
            st.success('Melhor resultado: ' + ', '.join(vencedores))
        if oov:
            st.caption('Termos fora do vocabulário: ' + ', '.join(oov))
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width='stretch')
        st.caption('Scores zero são mantidos para comparação; empates usam o ID do documento.')
    with pipeline:
        for i, texto in DOCUMENTOS.items():
            with st.expander(i, expanded=True):
                st.write(texto)
                st.write('Normalização:', normalizar(texto))
                st.write('Tokens finais:', tokens[i])
        st.write('Stopwords utilizadas:', sorted(STOPWORDS))
    with invertido:
        st.json(indice)
    with calculos:
        st.latex(r'TF(t,d)=\frac{f(t,d)}{|d|},\quad IDF(t)=\ln\frac{N+1}{df(t)+1}+1')
        st.latex(r'Score(d,q)=\sum_{t\in unique(q)\cap V}TF(t,d)\,IDF(t)')
        st.write('O bônus usa o cosseno dos vetores TF-IDF da consulta e dos documentos.')
        detalhes = []
        for i, ts in tokens.items():
            for t, frequencia in sorted(Counter(ts).items()):
                detalhes.append({'Documento': i, 'Termo': t, 'Frequência': frequencia,
                                 'TF': frequencia/len(ts), 'DF': len(indice[t]), 'IDF': idf[t],
                                 'TF-IDF': vetores[i][t]})
        st.dataframe(pd.DataFrame(detalhes), hide_index=True, width='stretch')

if __name__ == '__main__':
    main()
