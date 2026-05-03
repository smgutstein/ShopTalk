
import nltk
import re
import spacy

from nltk.corpus import stopwords
from tqdm import tqdm


def load_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from e
    

class Preprocessor():

    def __init__(self, item_id_dict):
        # Load spaCy model for NLP tasks
        self.nlp = load_spacy_model("en_core_web_sm")

        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))

        self.item_id_dict = item_id_dict

    # Define preprocessing functions
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Normalize case
        return text

    def tokenize_text(self, text):
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def remove_stopwords(self, tokens):
        return [word for word in tokens
                if word.lower() not in self.stop_words]

    def lemmatize_text(self, tokens):
        doc = self.nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]

    def chunk_text(self, text, chunk_size=512):
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) 
                for i in range(0, len(words), chunk_size)]

    def num_chunked_words(self, chunked_text):
        wc = 0
        for chunk_list in chunked_text:
            for chunk in chunk_list:
                wc += len(chunk.split())
        return wc

    # Load documents from directory
    def preprocess_documents(self):

        for curr_item in tqdm(self.item_id_dict.keys()):
            #for curr_item in self.item_id_dict.keys():
            text = self.item_id_dict[curr_item]['llm_str']
            cleaned_text = self.clean_text(text)
            tokenized_text = self.tokenize_text(cleaned_text)
            lemmatized_text = [self.lemmatize_text(self.remove_stopwords(sent.split()))
                                for sent in tokenized_text]

            chunked_text = [self.chunk_text(" ".join(sent)) 
                                    for sent in lemmatized_text
                                    if len(sent) > 0]
            self.item_id_dict[curr_item]['preproc_llm_str'] = chunked_text
            self.item_id_dict[curr_item]['word_count'] = self.num_chunked_words(chunked_text)

        return self.item_id_dict
