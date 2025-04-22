import os
import numpy as np
import math
from bs4 import BeautifulSoup
from collections import defaultdict
import re

DOCS_PATH = 'information-search/task_1/pages'
TFIDF_PATH = 'information-search/task_4/tfidf_output'
INDEX_PATH = 'information-search/task_3/inverted_index.txt'
NUM_DOCS = 111

def load_inverted_index():
    index = defaultdict(list)
    with open(INDEX_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            term, docs_str = line.strip().split(': ')
            docs = list(map(int, re.findall(r'\d+', docs_str)))
            index[term] = docs
    return index

def load_vocab_and_doc_vectors():
    vocab = {}
    doc_vectors = []
    for i in range(NUM_DOCS):
        vector = {}
        file_path = os.path.join(TFIDF_PATH, f'tfidf_doc_{i}_tokens.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                token, idf, tfidf = line.strip().split()
                vector[token] = float(tfidf)
                if token not in vocab:
                    vocab[token] = len(vocab)
        doc_vectors.append(vector)
    return vocab, doc_vectors

def load_idf():
    idf_values = {}
    for i in range(NUM_DOCS):
        file_path = os.path.join(TFIDF_PATH, f'tfidf_doc_{i}_tokens.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                token, idf, _ = line.strip().split()
                idf_values[token] = float(idf)
    return idf_values

def vectorize(tokens, vocab, idf_values):
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    total = len(tokens)
    vec = np.zeros(len(vocab))
    for token, count in tf.items():
        if token in vocab and token in idf_values:
            tfidf = (count / total) * idf_values[token]
            vec[vocab[token]] = tfidf
    return vec

def get_document_vector(doc_vec, vocab):
    vec = np.zeros(len(vocab))
    for token, tfidf in doc_vec.items():
        if token in vocab:
            vec[vocab[token]] = tfidf
    return vec

def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def extract_title_from_html(doc_id):
    path = os.path.join(DOCS_PATH, f'page_{doc_id}.html')
    try:
        with open(path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            title = soup.title.string if soup.title else f"Документ {doc_id}"
            return title.strip()
    except Exception as e:
        print(f"Ошибка при чтении {doc_id}: {e}")
        return f"Документ {doc_id}"

def search(query_tokens, top_n=10, return_results=False):
    vocab, doc_vectors = load_vocab_and_doc_vectors()
    idf_values = load_idf()
    index = load_inverted_index()

    query_vector = vectorize(query_tokens, vocab, idf_values)

    relevant_docs = set()
    for token in query_tokens:
        relevant_docs.update(index.get(token, []))

    scores = []
    for doc_id in relevant_docs:
        doc_vector = get_document_vector(doc_vectors[doc_id], vocab)
        score = cosine_similarity(query_vector, doc_vector)
        if score > 0:
            scores.append((doc_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for doc_id, score in scores[:top_n]:
        title = extract_title_from_html(doc_id)
        results.append({
            'doc_id': doc_id,
            'title': title,
            'score': round(score, 4)
        })
    
    if return_results:
        return results
    else:
        for r in results:
            print(f"{r['title']} (doc_{r['doc_id']}.txt) — Score: {r['score']}")



