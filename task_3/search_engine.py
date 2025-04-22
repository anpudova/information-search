import os
import re
from collections import defaultdict
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords, words

# Загрузка ресурсов NLTK
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('words')

stop_words = set(stopwords.words('english'))
english_vocab = set(words.words())

class BooleanSearchEngine:

    def __init__(self, documents):
        self.documents = documents
        self.index = self.build_inverted_index(documents)
        self.all_docs = set(range(len(documents)))
        self.save_index_to_file("information-search/task_3/inverted_index.txt")

    def is_english_word(self, word):
        return word.lower() in english_vocab

    def clean_tokens(self, tokens):
        return {
            t.lower()
            for t in tokens
            if t.isalpha()
            and len(t) > 1
            and self.is_english_word(t)
            and t.lower() not in stop_words
        }

    def tokenize_and_clean(self, text):
        tokens = nltk.word_tokenize(text.lower())
        return self.clean_tokens(tokens)

    def build_inverted_index(self, documents):
        index = defaultdict(set)
        for doc_id, text in enumerate(documents):
            for token in self.tokenize_and_clean(text):
                index[token].add(doc_id)
        return index

    def parse_query(self, query):
        def repl(token):
            token = token.strip()
            upper_token = token.upper()

            if upper_token in ('AND', 'И'):
                return '&'
            elif upper_token in ('OR', 'ИЛИ'):
                return '|'
            elif upper_token in ('NOT', 'НЕ'):
                return 'self.all_docs -'
            elif token == '(' or token == ')':
                return token
            else:
                cleaned = next(iter(self.tokenize_and_clean(token)), token.lower())
                return f"self.index.get('{cleaned}', set())"

        tokens = re.findall(r'\(|\)|\w+|AND|OR|NOT|И|ИЛИ|НЕ', query)
        parsed = ' '.join(repl(token) for token in tokens)
        return parsed

    def save_index_to_file(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for term in sorted(self.index):
                doc_ids = sorted(self.index[term])
                f.write(f"{term}: {doc_ids}\n")

    def search(self, query):
        try:
            parsed_query = self.parse_query(query)
            result = eval(parsed_query)
            return result
        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")
            return set()

def load_html_documents_from_folder(folder_path):
    documents = {}
    filenames = os.listdir(folder_path)
    for filename in filenames:
        if filename.endswith(".html"):
            match = re.search(r'\d+', filename)
            if match:
                doc_id = int(match.group())
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                    soup = BeautifulSoup(file, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    documents[doc_id] = text
    return documents

# Загрузка документов
folder_path = "information-search/task_1/pages"
documents_dict = load_html_documents_from_folder(folder_path)

documents = [documents_dict[k] for k in sorted(documents_dict.keys())]
doc_id_map = {i: k for i, k in enumerate(sorted(documents_dict.keys()))}

# Инициализация поисковой системы
engine = BooleanSearchEngine(documents)

# Интерфейс пользователя
print("Введите булев запрос (используйте AND, OR, NOT). Для выхода введите 'exit'.\n")
while True:
    query = input("Запрос: ").strip()
    if query.lower() == 'exit':
        print("Выход из программы.")
        break

    result = engine.search(query)

    if result:
        print("\nНайдены документы:")
        for idx in sorted(result):
            doc_num = doc_id_map[idx]  # Преобразуем индекс обратно в номер документа
            print(f"{doc_num}: {documents[idx][:200]}...")
    else:
        print("\nНичего не найдено.")
    print("-" * 50)