import numpy
import pandas
import re
from string import punctuation
import nltk
from nltk.stem.snowball import SnowballStemmer
englishStemmer=SnowballStemmer("english")

class Embeddings:
    
    def __init__(self):
        self.__wpt = nltk.WordPunctTokenizer()
        self.__stop_words = nltk.corpus.stopwords.words('english')
        self.__remove_terms = punctuation + '0123456789'

    def __split_camel_case_token(self, token):
        return re.sub('([a-z])([A-Z])', r'\1 \2', token).split()

    def __clean_punctuation(self, token):
        remove_terms = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
        cleaned = token
        for p in remove_terms:
            cleaned = cleaned.replace(p, ' ')
        return cleaned.split()

    def __clean(self, token):
        to_return = self.__clean_punctuation(token)
        new_tokens = []
        for t in to_return:
            new_tokens += self.__split_camel_case_token(t)
        to_return = new_tokens
        return to_return


    def __normalize_document(self, doc):
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
        doc = doc.lower()
        doc = doc.strip()
        # tokenize document
        tokens = self.__wpt.tokenize(doc)
        #Filter stopwords out of document
        filtered_tokens = [token for token in tokens if token not in self.__stop_words]
        #Filtering Stemmings
        filtered_tokens = [englishStemmer.stem(token) for token in filtered_tokens]
        #Filtering remove-terms
        filtered_tokens = [token for token in filtered_tokens if token not in self.__remove_terms and len(token)>2]
        # re-create document from filtered tokens
        return filtered_tokens

    def preprocess(self, sentence, vocab_set=None):
        tokens = sentence.split()
        new_tokens = []
        for token in tokens:
            new_tokens += self.__clean(token)
        tokens = new_tokens

        tokens = self.__normalize_document(' '.join(tokens))

        return tokens

    def get_embeddings_dict(self, embeddings_filename):
        embeddings_df = pandas.read_csv(embeddings_filename)
        embeddings_dict = dict()
        for col in list(embeddings_df)[1:]:
            embeddings_dict[col] = list(embeddings_df[col])
        return embeddings_dict

    def vectorize(self, sentence, embeddings_dict):
        processed_sentence = self.preprocess(sentence)

        matrix = []
        for token in processed_sentence:
            if token in embeddings_dict:
                matrix.insert(0, embeddings_dict[token])
        return numpy.matrix(matrix)

#sentence = "AAA AAA xxx BBB yyy CCC"
#embeddings = Embeddings()
#embeddings_dict = embeddings.get_embeddings_dict('test.csv')
#print(embeddings_dict)
#print(vectorize(sentence, embeddings_dict))
