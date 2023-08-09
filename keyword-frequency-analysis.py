import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

nltk.download("punkt")
nltk.download("stopwords")

def process_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove punctuation and non-alphanumeric characters
    tokens = [token for token in tokens if token.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def plot_frequency_distribution(tokens):
    fdist = FreqDist(tokens)
    fdist.plot(30, cumulative=False)
    plt.show()

if __name__ == "__main__":
    text = """
your text
    """
    
    processed_tokens = process_text(text)
    plot_frequency_distribution(processed_tokens)
