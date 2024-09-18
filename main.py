from flask import Flask, url_for, render_template, request, send_file
import pickle
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pickle
import io
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)


nlp = spacy.load("en_core_web_sm")

model = pickle.load(open('my_model.pkl', 'rb'))

def preprocess(txt):
    doc = nlp(txt)
    filtered_token = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            filtered_token.append(token.lemma_.lower())
    return " ".join(filtered_token)


@app.route('/')
def Home():
    return render_template('sentiment.html')



@app.route('/submit', methods=['POST'])
def submit():
    label_mapping = {0: "Positive", 1: "Negative"}  # Example mapping
    text = request.form["review"]
    preprocessed_text = preprocess(text)
    prediction = model.predict([preprocessed_text])
    probabilities = model.predict_proba([preprocessed_text])  # This gives probabilities for each class
    predicted_label = label_mapping[prediction[0]]

    positive_prob = probabilities[0][0] * 100  # Probability for positive class
    negative_prob = probabilities[0][1] * 100  # Probability for negative class

    return render_template("sentiment.html",Predicted = predicted_label,
        Positive = positive_prob, 
        Negative = negative_prob,
        submitted = True
 )

@app.route('/chart')
def chart():
    # Get the positive and negative probabilities from query parameters
    positive = float(request.args.get('positive', 0))
    negative = float(request.args.get('negative', 0))
    
    # Create the bar chart using Matplotlib
    labels = ['Positive','Negative']
    values = [positive, negative]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['green','red'])
    ax.set_ylabel('Confidence (%)')
    ax.set_ylim([0, 100])

    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Return the image as a response
    return send_file(img, mimetype='image/png')







if __name__ == '__main__':
    app.run(debug=True)
