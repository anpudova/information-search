import re
from collections import defaultdict
import pymorphy2

class BooleanSearchEngine:
    def __init__(self, documents):
        self.morph = pymorphy2.MorphAnalyzer()
        self.documents = documents
        self.index = self.build_inverted_index(documents)
        self.all_docs = set(range(len(documents)))
        self.save_index_to_file("inverted_index.txt")

    def lemmatize(self, word):
        return self.morph.parse(word)[0].normal_form

    def tokenize(self, text):
        words = re.findall(r'\w+', text.lower())
        return [self.lemmatize(word) for word in words]

    def build_inverted_index(self, documents):
        index = defaultdict(set)
        for doc_id, text in enumerate(documents):
            for word in self.tokenize(text):
                index[word].add(doc_id)
        return index

    def parse_query(self, query):
        def repl(token):
            token = token.strip()
            if token.upper() == 'AND':
                return '&'
            elif token.upper() == 'OR':
                return '|'
            elif token.upper() == 'NOT':
                return 'self.all_docs -'
            elif token == '(' or token == ')':
                return token
            else:
                lemma = self.lemmatize(token.lower())
                return f"self.index.get('{lemma}', set())"

        tokens = re.findall(r'\(|\)|\w+|AND|OR|NOT', query)
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

# Документы
documents = [
    "Программист написал сложный алгоритм для анализа текста.",
    "Программирование требует усидчивости и логического мышления.",
    "Алгоритмы могут быть простыми или очень сложными, в зависимости от задачи.",
    "Обработка информации — ключевой этап в большинстве программ.",
    "Основная задача — разработать программный код.",
    "Был проведен анализ данного текста с помощью эффективного программного кода.",
]

engine = BooleanSearchEngine(documents)

# Цикл запроса
print("Введите булев запрос (используйте AND, OR, NOT). Для выхода введите 'exit'.\n")
while True:
    query = input("Запрос: ").strip()
    if query.lower() == 'exit':
        print("Выход из программы.")
        break

    result = engine.search(query)

    if result:
        print("\nНайдены документы:")
        for doc_id in sorted(result):
            print(f"{doc_id}: {documents[doc_id]}")
    else:
        print("\nНичего не найдено.")
    print("-" * 50)