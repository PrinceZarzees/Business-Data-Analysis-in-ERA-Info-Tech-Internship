import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import sys
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize tokenizer and lemmatizer
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Get the list of stopwords
stopwords_list = stopwords.words('english')

# Define the exceptions
exceptions = {'eftn', 'ft', 'bkash', 'nogod', 'rtgs', 'pos',
              'cib', 'paywell', 'challan', 'npsb', 'dps',
              'atm', 'trf', 'sonod'}

# Function to lemmatize text
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]

# Function to preprocess the text
def preprocess_text(text):
    # Remove special characters and non-alphabetic characters
    text = re.sub(r"[^a-zA-Z ]+", "", text)
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word.lower() not in stopwords_list])
    # Lemmatize the text
    text = ' '.join(lemmatize_text(text))
    return text
def determine_cluster(text,method):
  if method=="kmeans":
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_loaded = pickle.load(f)
    with open('vectorizer_kmeans.pkl', 'rb') as f:
        vectorizer_loaded = pickle.load(f)
  elif method=="lda":
    with open('lda_model.pkl', 'rb') as f:
        lda_loaded = pickle.load(f)
    with open('vectorizer_lda.pkl', 'rb') as f:
        vectorizer_loaded = pickle.load(f)
        # Transform the input text using the loaded vectorizer
  else:
    return "error"
  text_vectorized = vectorizer_loaded.transform([text])
  if method=="kmeans":
  # Predict the cluster using the loaded K-means model
    cluster = kmeans_loaded.predict(text_vectorized)[0]
  elif method=="lda":
    doc_topic_dist = lda_loaded.transform(text_vectorized)
    # Determine the dominant topic for the given text
    cluster = doc_topic_dist.argmax(axis=1)[0]
  else:
    return "error"
  return cluster
  

  # Test the clustering function

if len(sys.argv) != 3:
      print("Usage: python clustering_script.py <text> <method>")
      exit()
    
text = sys.argv[1]
method = sys.argv[2]
preprocessed_text=preprocess_text(text);
# Determine the cluster using the specified method
cluster = determine_cluster(preprocessed_text, method)

# Print the preprocessed text and cluster
print("Preprocessed Text:", preprocessed_text)
print("Cluster:", cluster)