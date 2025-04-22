from flask import Flask, render_template, request
import re
from vector_search import search
from flask import send_file
from bs4 import BeautifulSoup
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        query = request.form['query']
        tokens = re.findall(r'\w+', query.lower())
        results = search(tokens, return_results=True)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
