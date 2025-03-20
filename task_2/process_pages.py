import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

STOPWORDS = set([
    "i", "me", "my", "we", "you", "he", "she", "it", "they", "a", "an", "the", 
    "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", 
    "by", "for", "with", "about", "against", "between", "into", "through", "during", 
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", 
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", 
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", 
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", 
    "now"
])

INPUT_FOLDER = "pages"
OUTPUT_TOKENS = "tokens.txt"
OUTPUT_LEMMAS = "lemmas.txt"

def clear_files():
    open(OUTPUT_TOKENS, 'w', encoding='utf-8').close()
    open(OUTPUT_LEMMAS, 'w', encoding='utf-8').close()

def is_english_word(word):
    return bool(re.match(r'^[a-zA-Z]+$', word))

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

def clean_tokens(tokens):
    return {t.lower() for t in tokens if t.isalpha() and is_english_word(t) and t.lower() not in STOPWORDS}

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
    lemmas = {}
    for token in tokens:
        pos = get_wordnet_pos(token)
        lemma = lemmatizer.lemmatize(token, pos)  # Лемматизация с учетом части речи
        if lemma not in lemmas:
            lemmas[lemma] = set()
        lemmas[lemma].add(token.lower())

    for lemma in lemmas:
        lemmas[lemma] = " ".join(sorted(lemmas[lemma]))

    return lemmas

def process_files():
    all_tokens = set()

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".html"):
            with open(os.path.join(INPUT_FOLDER, filename), "r", encoding="utf-8") as file:
                html_content = file.read()
                text = extract_text_from_html(html_content)
                tokens = word_tokenize(text)  # Токенизация
                clean_tokens_list = clean_tokens(tokens)  # Фильтрация
                all_tokens.update(clean_tokens_list)  # Добавляем в общий список

    # Лемматизация
    lemmas = lemmatize_tokens(all_tokens)
    clear_files()

    # Запись
    with open(OUTPUT_TOKENS, "w", encoding="utf-8") as f:
        for token in sorted(all_tokens):
            f.write(f"{token}\n")

    with open(OUTPUT_LEMMAS, "w", encoding="utf-8") as f:
        for lemma, words in sorted(lemmas.items()):
            f.write(f"{lemma}: {words}\n")

    print(f"Файлы сохранены: {OUTPUT_TOKENS}, {OUTPUT_LEMMAS}")

process_files()

