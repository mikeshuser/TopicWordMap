# -*- coding : utf-8 -*-
"""
Convenience module for pre-processing text for NLP
Some funcs depend on spacy and textblob libraries

2020-07-24 : Most of these functions were created for machine learning. 
    However, with modern Transformer architectures, most of these text 
    pre-processing steps are no longer necessary.
"""

__author__ = "Mike Shuser"

import re
from textblob import TextBlob

punct_and_sym = [',', '.', "'", '"', ':', ')', '(', '-', '|', ';', "'", '$', 
    '&', '/', '[', ']', '>', '%', '=', '*', '+', '\\', '•',  
    '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  
    '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', 
    '′', '█', '½', '…', '“', '★', '”', '–', '●', 
    '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', 
    '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', 
    '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', 
    '♫', '☆', '¯', '♦', '¤', '▲', '¸', '¾', 
    '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', 
    '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 
    'ï', 'Ø', '¹', '≤', '‡', '√', '?', '!', '#']

contraction_dict = {
    "ain't" : "is not", 
    "aren't" : "are not",
    "can't" : "cannot", 
    "could've" : "could have", 
    "couldn't" : "could not", 
    "didn't" : "did not",  
    "doesn't" : "does not", 
    "don't" : "do not", 
    "hadn't" : "had not", 
    "hasn't" : "has not", 
    "haven't" : "have not", 
    "he'd" : "he would",
    "he'll" : "he will", 
    "he's" : "he is", 
    "she'd" : "she would", 
    "she'll" : "she will", 
    "she's" : "she is", 
    "how'd" : "how did", 
    "how'll" : "how will", 
    "how's" : "how is",  
    "i'd" : "i would", 
    "i'll" : "i will",  
    "i'm" : "i am", 
    "i've" : "i have", 
    "isn't" : "is not", 
    "it'd" : "it would", 
    "it'll" : "it will", 
    "it's" : "it is", 
    "let's" : "let us", 
    "mayn't" : "may not", 
    "might've" : "might have",  
    "mustn't" : "must not", 
    "needn't" : "need not", 
    "shan't" : "shall not", 
    "should've" : "should have", 
    "shouldn't" : "should not", 
    "that'd" : "that would", 
    "that's" : "that is", 
    "here's" : "here is",
    "there's" : "there is",
    "they'd" : "they would", 
    "they'll" : "they will", 
    "they're" : "they are", 
    "they've" : "they have", 
    "wasn't" : "was not", 
    "we'd" : "we would", 
    "we'll" : "we will", 
    "we're" : "we are", 
    "we've" : "we have", 
    "weren't" : "were not", 
    "what'll" : "what will", 
    "what're" : "what are",  
    "what's" : "what is", 
    "when's" : "when is", 
    "where'd" : "where did", 
    "where's" : "where is", 
    "who'll" : "who will", 
    "who's" : "who is", 
    "why've" : "why have", 
    "won't" : "will not", 
    "would've" : "would have", 
    "wouldn't" : "would not", 
    "y'all" : "you all", 
    "you'd" : "you would", 
    "you'll" : "you will", 
    "you're" : "you are", 
    "you've" : "you have",
}

encoding_symbols = [
    "\x84", 
    "\x85", 
    "\x90", 
    "\x91", 
    "\x93", 
    "\x94", 
    "\x95", 
    "\x96", 
    "\x97", 
    "\x98", 
    "\x99",
]
                
def clean_puncts(raw_text):

    """
    Simple removal of all punctuation and symbols from punct_and_sym
    """

    text = str(raw_text)
    for punct in punct_and_sym:
        if punct in text:
            text = text.replace(punct, ' ')
    return text

def expand_contractions(raw_text):

    """
    Expand all contractions in contraction_dict
    """

    text = str(raw_text).lower()
    for k, v in contraction_dict.items():
        text = re.sub(k, v, text)
    return text

def remove_special_chars(raw_text):

    """
    Remove certain special characters if French encoding causing trouble
    """

    text = str(raw_text)
    for sym in encoding_symbols:
        text = re.sub(r"{}".format(sym), " ", text)
    return text

def norm_apostrophe(raw_text):

    """
    Normalize apostrophes to standard form
    """

    text = str(raw_text)
    text = re.sub(r"’", "'", text)
    text = re.sub(r"`", "'", text)
    return text

def remove_html(raw_text):

    """
    Remove html tags
    """

    text = str(raw_text)
    cleaner = re.compile('<.*?>')
    text = re.sub(cleaner, '', text)
    return text

def remove_non_alpha(raw_text):

    """
    Remove all non-letters
    """

    text = str(raw_text).lower()
    text = re.sub("[^a-z]", " ", text)
    return text
    
def replace_nums(raw_text, remove=False):

    """
    Either replace all numbers with # placeholders of varying length,
    or convert numbers to words
    """

    text = str(raw_text)
    if remove:
        text = re.sub('[0-9]{5,}', '#####', text)
        text = re.sub('[0-9]{4}', '####', text) 
        text = re.sub('[0-9]{3}', '###', text)
        text = re.sub('[0-9]{2}', '##', text)
        text = re.sub('[0-9]', '#', text)
    else:
        text = re.sub(r"0", "zero", text)
        text = re.sub(r"1", "one", text)
        text = re.sub(r"2", "two", text)
        text = re.sub(r"3", "three", text)
        text = re.sub(r"4", "four", text)
        text = re.sub(r"5", "five", text)
        text = re.sub(r"6", "six", text)
        text = re.sub(r"7", "seven", text)
        text = re.sub(r"8", "eight", text)
        text = re.sub(r"9", "nine", text)
        text = re.sub(r"10", "ten", text)
    return text

def delete_nums(raw_text, remove=False):

    """
    delete all numeric values
    """

    text = str(raw_text)
    text = re.sub('[0-9]', ' ', text)
    return text

def lemmatize(raw_text: str, spacy_nlp, exceptions=[]):

    """
    lemmatize all words in string, skipping over optional exceptions list
    requires nlp object from spacy module
    """

    despaced = ' '.join(raw_text.split())
    lemmas = []
    for word in spacy_nlp(despaced):
        if word.text in exceptions:
            lemmas.append(word.text)
        else:
            lemmas.append(word.lemma_)
    return ' '.join(lemmas)

def remove_stop(raw_text:str, spacy_nlp):

    """
    remove all stop words from string
    requires nlp object from spacy module
    """

    cleaned = [word.text for word in spacy_nlp(raw_text) if not word.is_stop]  
    return ' '.join(cleaned)

def is_stop_word(word: str, spacy_nlp):

    """
    check if word is a stop word
    requires nlp object from spacy module
    """

    return spacy_nlp(word)[0].is_stop

def get_pos(text: str, single_word:bool, spacy_nlp):

    """
    get parts of speech from a string. Retuns a list
    requires nlp object from spacy module
    """

    if single_word:
        pos = spacy_nlp(text)[0].tag_
    else:
        pos = [token.tag_ for token in spacy_nlp(text)]
    return pos
    
def singularize(raw_text: str, spacy_nlp):

    """
    normalize all plural words to be singular
    requires nlp object from spacy module
    """

    converted = raw_text
    for token in spacy_nlp(raw_text):
        if token.tag_ == 'NNS':
            converted = converted.replace(token.text, 
                TextBlob(token.text).words.singularize()[0])
    
    return converted

def translate_french(
    fr_responses: list, 
    project_id: str, 
    key_path: str, 
    batch_size=128,
):

    """
    Use google cloud api to batch translate french to english
    Assumes you have a project_id registered on google cloud
    key_path needs to be a json file with your credentials
    """

    from google.cloud import translate
    from math import ceil
    import os
    
    print("Translating French to English")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
    client = translate.TranslationServiceClient()
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"
    batches = ceil(len(fr_responses)/batch_size)
    english = []
    left_slice = 0
    
    for batch in range(batches):
        print(f"requesting {left_slice}:{batch_size + left_slice}")
        text = fr_responses[left_slice : batch_size + left_slice]
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": text,
                "mime_type": "text/plain",  # mime types: text/plain, text/html
                "source_language_code": "fr",
                "target_language_code": "en",
            }
        )

        for translation in response.translations:
            english.append(translation.translated_text)
        
        left_slice += batch_size
    return english
