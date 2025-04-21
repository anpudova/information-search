import os
import re
import math
from collections import Counter
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, words, stopwords

lemmatizer = WordNetLemmatizer()
english_vocab = set(words.words())
stop_words = set(stopwords.words('english'))

# Загрузка всех документов
def load_html_documents_from_folder(folder_path):
    documents = {}
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(".html")])
    for filename in filenames:
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            text = soup.get_text()
            doc_id = int(re.search(r'\d+', filename).group())
            documents[doc_id] = text
    return documents

def is_english_word(word):
    return word.lower() in english_vocab

def clean_tokens(tokens):
    return {
        t.lower()
        for t in tokens
        if t.isalpha()
        and len(t) > 1
        and is_english_word(t)
        and t.lower() not in stop_words
    }

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1]
    if tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.ADJ

def lemmatize_tokens(tokens):
    lemmas = []
    for token in tokens:
        if token.isalpha():
            pos = get_wordnet_pos(token)
            lemma = lemmatizer.lemmatize(token, pos)  # Лемматизация с учетом части речи
            lemmas.append(lemma.lower())
    return lemmas

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    clean_tokens_list = clean_tokens(tokens)
    lemmas = lemmatize_tokens(clean_tokens_list)
    return clean_tokens_list, lemmas

# Подсчет TF
def compute_tf(tokens):
    tf = Counter(tokens)
    total_tokens = len(tokens)
    for token in tf:
        tf[token] /= total_tokens
    return tf

# Подсчет IDF
def compute_idf(documents, all_tokens):
    idf = {}
    total_docs = len(documents)
    for token in all_tokens:
        docs_with_term = sum(1 for doc in documents if token in doc)
        idf[token] = math.log(total_docs / docs_with_term)
    return idf

# Подсчет TF-IDF
def compute_tfidf(tf, idf):
    tfidf = {}
    for token, tf_value in tf.items():
        tfidf[token] = tf_value * idf.get(token, 0)
    return tfidf

# Сохранение результатов в файлы
def save_tfidf_to_file(doc_id, tfidf_tokens, tfidf_lemmas, idf_terms, idf_lemmas, output_dir):
    # Сохранение для токенов
    with open(os.path.join(output_dir, f'tfidf_doc_{doc_id}_tokens.txt'), 'w', encoding='utf-8') as file:
        for token in sorted(tfidf_tokens.keys()):
            tfidf_value = tfidf_tokens[token]
            idf_value = idf_terms.get(token, 0)
            file.write(f"{token} {idf_value} {tfidf_value}\n")
    
    # Сохранение для лемм
    with open(os.path.join(output_dir, f'tfidf_doc_{doc_id}_lemmas.txt'), 'w', encoding='utf-8') as file:
        for lemma in sorted(tfidf_lemmas.keys()):
            tfidf_value = tfidf_lemmas[lemma]
            idf_value = idf_lemmas.get(lemma, 0)
            file.write(f"{lemma} {idf_value} {tfidf_value}\n")

def load_terms_and_lemmas(doc_id, tokens_folder, lemmas_folder):
    # Чтение терминов из файлов для каждого документа
    with open(os.path.join(tokens_folder, f"tokens_{doc_id}.txt"), "r", encoding="utf-8") as file:
        tokens = set(file.read().splitlines())
    
    # Чтение лемм из файлов
    lemmas = {}
    with open(os.path.join(lemmas_folder, f"lemmas_{doc_id}.txt"), "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(":")
            if len(parts) == 2:
                lemma = parts[0].strip()
                lemma_tokens = parts[1].strip().split()
                lemmas[lemma] = set(lemma_tokens)
    
    return tokens, lemmas

def process_documents(input_folder, output_folder, tokens_folder, lemmas_folder):
    documents = load_html_documents_from_folder(input_folder)
    
    all_tokens = set()
    all_lemmas = set()
    
    documents_lemmas = [] 
    documents_tokens = [] 
    doc_ids = sorted(documents.keys())
    
    for doc_id in doc_ids:
        tokens, lemmas = load_terms_and_lemmas(doc_id, tokens_folder, lemmas_folder)
        documents_lemmas.append(set(lemmas.keys()))
        documents_tokens.append(set(tokens))
        all_tokens.update(tokens)
        all_lemmas.update(lemmas.keys())
    
    idf_tokens = compute_idf(documents_tokens, all_tokens)
    idf_lemmas = compute_idf(documents_lemmas, all_lemmas)
    
    for doc_id in doc_ids:
        text = documents[doc_id]
        tokens, lemmatized_tokens = preprocess_text(text)
        tf_tokens = compute_tf(tokens)
        tf_lemmas = compute_tf(lemmatized_tokens)
        tfidf_tokens = compute_tfidf(tf_tokens, idf_tokens)
        tfidf_lemmas = compute_tfidf(tf_lemmas, idf_lemmas)
        
        save_tfidf_to_file(doc_id, tfidf_tokens, tfidf_lemmas, idf_tokens, idf_lemmas, output_folder)

input_folder = "information-search/task_1/pages"  # Папка с HTML документами
output_folder = "information-search/task_4/tfidf_output"  # Папка для результатов
tokens_folder = "information-search/task_2/tokens"  # Папка с терминами для каждого документа
lemmas_folder = "information-search/task_2/lemmas"  # Папка с леммами для каждого документа

# Создание папки для сохранения результатов
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Запуск обработки документов
process_documents(input_folder, output_folder, tokens_folder, lemmas_folder)
