from flask import Flask, request, render_template, jsonify
import pickle
import re
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__, template_folder="templates")

# Download NLTK stopwords dataset
import nltk
nltk.download('stopwords')

# Load your vectorizer and model
with open('TfidfVectorizer.pickle', 'rb') as file:
    vectorizer = pickle.load(file)

with open('MultinomialNB.pickle', 'rb') as file:
    model = pickle.load(file)

# Read stopwords from the file
with open('stopwords-ms.txt', 'r') as file:
    stopwords = set(word.strip() for word in file)
    
# Create Lemmatizer
word_lemmatizer = WordNetLemmatizer()

# Create Sastrawi stemmer
stemmer = StemmerFactory().create_stemmer()

def preprocess_and_stem(dataset):
    # Convert to lowercase
    dataset = dataset.lower()

    # Replace URLs with 'URL'
    dataset = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', 'URL', dataset)

    # Remove mentions
    dataset = re.sub(r'@[^\s]+', '', dataset)

    # Remove non-alphanumeric characters and extra spaces
    dataset = re.sub(r'[^a-zA-Zа-яА-Я1-9]+', ' ', dataset)
    dataset = re.sub(r' +', ' ', dataset)

    # Tokenize and lemmatize
    words = dataset.split()
    lemmatized_words = [word_lemmatizer.lemmatize(word) for word in words]
    
    # Remove stopwords
    filtered_words = [word for word in lemmatized_words if word.lower() not in stopwords]

    # Join the filtered words with a space between them
    processed_text = ' '.join(filtered_words)

    # Apply stemming
    stemmed_text = stemmer.stem(processed_text)

    return stemmed_text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Extract the input data from the JSON request
            input_data = request.json.get('user_input', '')

            # Preprocess and stem the input text
            processed_text = preprocess_and_stem(input_data)

            # Vectorize the processed text
            vectorized_text = vectorizer.transform([processed_text])

            # Make prediction using the loaded model
            prediction = model.predict(vectorized_text)

            # Return the prediction as JSON response
            return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/')
def home():
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
