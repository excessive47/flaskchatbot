from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import cosine
from utils.embeddings_utils import get_embedding

app = Flask(__name__)

# Lade die Daten beim Start der App
filepath = "data/fragen_und_antworten_embeddings.csv"
df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')

def string_to_array(s):
    return np.array(ast.literal_eval(s))


# Konvertiere die Einbettungsstrings in Arrays
df['embedding'] = df['embedding'].apply(string_to_array)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['question']
        user_embedding = get_embedding(user_input)
        df['similarity'] = df['embedding'].apply(lambda x: 1 - cosine(x, user_embedding))
        max_similarity_index = df['similarity'].idxmax()
        most_similar_question = df.loc[max_similarity_index]['Frage']
        most_similar_answer = df.loc[max_similarity_index]['Antwort']
        similarity_score = df.loc[max_similarity_index]['similarity']
        return render_template('index.html', question=user_input, similar_question=most_similar_question, similar_answer=most_similar_answer, similarity=similarity_score)
    return render_template('index.html', question=None)

if __name__ == '__main__':
    app.run(debug=True)
