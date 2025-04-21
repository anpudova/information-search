import os
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, words, stopwords

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('words')
#nltk.download('omw-1.4')
#nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
english_vocab = set(words.words())
stop_words = set(stopwords.words('english'))

INPUT_FOLDER = "information-search/task_1/pages"
OUTPUT_TOKENS = "information-search/task_2/tokens/tokens"
OUTPUT_LEMMAS = "information-search/task_2/lemmas/lemmas"

def clear_files():
    open(OUTPUT_TOKENS, 'w', encoding='utf-8').close()
    open(OUTPUT_LEMMAS, 'w', encoding='utf-8').close()

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

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

    os.makedirs("information-search/task_2/tokens", exist_ok=True)
    os.makedirs("information-search/task_2/lemmas", exist_ok=True)
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".html"):
            with open(os.path.join(INPUT_FOLDER, filename), "r", encoding="utf-8") as file:
                all_tokens = set()
                lemmas = {}
                html_content = file.read()
                text = extract_text_from_html(html_content)
                tokens = word_tokenize(text.lower())  # Токенизация
                clean_tokens_list = clean_tokens(tokens)  # Фильтрация
                all_tokens.update(clean_tokens_list)  # Добавляем в общий список
                lemmas = lemmatize_tokens(all_tokens)

                 # Запись
                with open(OUTPUT_TOKENS + "_" +  str(int(re.search(r'\d+', filename).group())) + ".txt", "w", encoding="utf-8") as f:
                    for token in sorted(all_tokens):
                        f.write(f"{token}\n")

                with open(OUTPUT_LEMMAS + "_" +  str(int(re.search(r'\d+', filename).group())) + ".txt", "w", encoding="utf-8") as f:
                    for lemma, words in sorted(lemmas.items()):
                        f.write(f"{lemma}: {words}\n")


process_files()

