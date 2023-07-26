
# Load the necessary libraries and perform preprocessing
import pandas as pd
from flask import Flask, render_template, request
from main import vect, logreg, data_processing


df = pd.read_csv('twitter_training.csv')


# Define the Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to render the sentiment analysis page
@app.route('/analyze' , methods=['POST'])
def analyze():
    return render_template('analyze.html')

# Define the route to handle the sentiment analysis form submission
@app.route('/result', methods=['POST'])
def result():
    text = request.form['text']

    # Perform sentiment analysis on the provided text
    processed_text = data_processing(text)
    vectorized_text = vect.transform([processed_text])
    sentiment = logreg.predict(vectorized_text)[0]

    return render_template('result.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
# * Running on http://127.0.0.1:5000/