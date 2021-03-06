# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_utils.ipynb (unless otherwise specified).

__all__ = ['Dynamic_Dataset', 'Processing_Dataset', 'Embeddings', 'englishStemmer']

# Cell
import random
import sys
import os.path
import zipfile
from pathlib import Path
import pandas as pd

"@danaderp May'20 Refactoring for enhancing time complexity with pandas vectorization"

class Dynamic_Dataset:
    """
    This class efficiently 'stores' a dataset. Only a list of filenames and
    mappings to their ground truth values are stored in memory. The file
    contents are only brought into memory when requested.

    This class supports indexing, slicing, and iteration.

    A user can treat an instance of this class exactly as they would a list.
    Indexing an instance of this class will return a tuple consisting of
    the ground truth value and the file content of the filename at that index.

    A user can request the filename at an index with get_id(index)

    Example:

        dataset = Dynamic_Dataset(ground_truth)

        print(dataset.get_id(0))
            -> gitlab_79.txt

        print(dataset[0])
            -> ('(1,0)', 'The currently used Rails version, in the stable ...

        for x in dataset[2:4]:
            print(x)
                -> ('(1,0)', "'In my attempt to add 2 factor authentication ...
                -> ('(1,0)', 'We just had an admin accidentally push to a ...

    """

    def __init__(self, ground_truth, path, isZip):
        '''
        @param ground_truth (dict): A dictionary mapping filenames to ground truth values
        @param path (list): The path to the directory containing the dataset
        @param isZip (bool): Whether or not the data is contained in a zip
        '''
        self.__keys = list(ground_truth.keys())
        self.__ground_truth = ground_truth
        self.__path = path
        self.__isZip = isZip

    # Retrieve the contents of the specified file
    def __get_issue(self, filename):
        '''
        @param filename (str): The name of the file whose contents should be read
        '''
        # If the data is stored in a zip file, find that file and get the specified file from it
        if self.__isZip:
            paths = [str(x) for x in Path(self.__path).glob("**/*.zip")]
            for onezipath in paths:
                archive = zipfile.ZipFile( onezipath, 'r')
                contents = archive.read('issues/' + filename)
        # Otherwise, open the file and read it right away
        else:
            with open(self.__path+'issues/' + filename, 'r') as file:
                contents = file.read()
        return contents.strip()


    def get_id(self, index):
        """Get the name of the file at the specified index"""
        return self.__keys[index]


    def __len__(self):
        """The number of files contained in the dataset"""
        return len(self.__keys)

    def __setitem__(self, key, item):
        raise ValueError

    # Return a tuple with the ground truth and file content at the specified index
    # If key is a slice, returns a dataset containing only the items at the specified indices
    def __getitem__(self, key):
        if type(key) == slice:
            new_keys = self.__keys[key.start:key.stop:key.step]
            new_gt = dict()
            for key in new_keys:
                new_gt[key] = self.__ground_truth[key]
            return Dynamic_Dataset(new_gt,self.__path,self.__isZip)
        else:
            id = self.__keys[key]
            return (self.__ground_truth[id], self.__get_issue(id))

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index < len(self.__keys):
            to_return = self[self.__index]
            self.__index += 1
            return to_return
        else:
            raise StopIteration

# Cell
# This comment is necessary for the nbdev doc links to build properly

class Processing_Dataset:
    """
    This class wraps up processing and will match issue our data corpus with it's ground truth.
    This class also creates our test train split.
    """

    def __init__(self, path):
        self.__path = path

    def get_issue(self, filename):
        """
        Give the contents of the specified file
        """
        with open(self.__path + filename, 'r') as file:
            contents = file.read()
        return contents.strip()

    def get_ground_truth(self):
        """
        Returns a dictionary representing the ground truth mapping
        to filenames in the path this class was initialized with
        """
        gt = dict()
        with open(self.__path+'full_ground_truth.txt') as gt_file:
            for line in gt_file.readlines():
                tokens = line.split()
                filename = tokens[0]
                security_status = tokens[1]
                if filename in gt:
                    raise KeyError("Invalid Ground Truth: Duplicate issue [{}]".format(filename))
                gt[filename] = security_status
        return gt

    def get_test_and_training(self, ground_truth, test_ratio=0.1, isZip = False):
        """
        Given the input ground truth dictionary, generate a train test split with the given ratio: default 0.1.
        If isZip is true, then we will attempt to read the data as a zip archive.  If not, we will try to read them normally.
        Returns a tuple of form (test, train), where test and train are of class Dynamic_Dataset
        """
        ids = list(ground_truth.keys())
        sr = []
        nsr = []

        for id in ids:
            if ground_truth[id] == '(1,0)':
                sr.append(id)
            elif ground_truth[id] == '(0,1)':
                nsr.append(id)
            else:
                raise ValueError("There was an issue with ground truth: {} - {}".format(id, ground_truth[id]))


        n_test = int(len(sr) * test_ratio)
        sr_test = random.sample(sr, n_test)
        nsr_test = random.sample(nsr, n_test)

        test_gt = dict()
        train_gt = dict(ground_truth)

        for i in range(n_test):
            sr.remove(sr_test[i])
            test_gt[sr_test[i]] = '(1,0)'
            del train_gt[sr_test[i]]

            nsr.remove(nsr_test[i])
            test_gt[nsr_test[i]] = '(0,1)'
            del train_gt[nsr_test[i]]

        test = Dynamic_Dataset(test_gt,self.__path, isZip)
        train = Dynamic_Dataset(train_gt,self.__path, isZip)

        return (test, train)

# Cell
import numpy
import pandas
import re
from string import punctuation
import nltk
from nltk.stem.snowball import SnowballStemmer
englishStemmer=SnowballStemmer("english")

class Embeddings:
    """
    Embeddings class is responsible for cleaning, normalizing, and vectorizing a given corpus.
    """
    def __init__(self):
        self.__wpt = nltk.WordPunctTokenizer()
        self.__stop_words = nltk.corpus.stopwords.words('english')
        self.__remove_terms = punctuation + '0123456789'

    # Splits a camel case token into a list of words
    def __split_camel_case_token(self, token):
        return re.sub('([a-z])([A-Z])', r'\1 \2', token).split()

    # Splits token into a list of substrings delimited by punctuation
    def __clean_punctuation(self, token):
        remove_terms = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
        cleaned = token
        for p in remove_terms:
            cleaned = cleaned.replace(p, ' ')
        return cleaned.split()

    # Combines clean_punctuation and split_camel_case_token
    def __clean(self, token):
        cleaned_tokens = self.__clean_punctuation(token)
        new_tokens = []
        for t in cleaned_tokens:
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
        """
        Preprocess a given sentence string by cleaning each token, and normalizing.
        We tokenize, filter stopwords, filter stemmings, and filter remove-terms.
        Returns a list of tokens.
        """
        tokens = sentence.split()
        new_tokens = []
        for token in tokens:
            new_tokens += self.__clean(token)
        tokens = new_tokens

        tokens = self.__normalize_document(' '.join(tokens))

        return tokens

    def get_embeddings_dict(self, embeddings_filename):
        """
        Return a dictionary representation of the embeddings from the given embeddings csv file located at the input path
        """
        embeddings_df = pandas.read_csv(embeddings_filename)
        embeddings_dict = dict()
        for col in list(embeddings_df)[1:]:
            embeddings_dict[col] = list(embeddings_df[col])
        return embeddings_dict

    def vectorize(self, sentence, embeddings_dict):
        """
        Takes an input sentence as a string, preprocesses it, then vectorizes it based on the input
        embeddings dictionary.
        Returns a numpy matrix representing the vectorized sentence
        """
        processed_sentence = self.preprocess(sentence)

        matrix = []
        for token in processed_sentence:
            if token in embeddings_dict:
                matrix.insert(0, embeddings_dict[token])
        return numpy.matrix(matrix)