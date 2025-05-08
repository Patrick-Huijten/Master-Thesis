import time
import fitz
import pandas as pd
import os
import string
import spacy
from spacy.util import is_package
from spacy.cli import download

# Dummy classifier training function, remove once the final version is ready
def train_classifier():
    time.sleep(3)  # Simulate time-consuming training
    return "Model trained successfully!"

#________________________________________________________________________________

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def case_normalization(text):
    """Returns string of input containing only lowercase letters apart from [NEWLINE], which replaces \n"""
    text = text.lower()
    text = text.replace('\n', ' [NEWLINE] ')
    while text != text.replace('  ', ' '):
        text = text.replace('  ', ' ')
    return text

def remove_punctuation(text):
    """Returns the input text with all punctuation removed"""
    
    text = text.translate(text.maketrans("", "", string.punctuation))
    text = text.replace("NEWLINE", "[NEWLINE]")
    return text

def remove_stopwords(text, nlp):
    """Returns string of input text with stopwords removed"""
    
    doc = nlp(text)
    filtered_words = [token.text for token in doc if not token.is_stop]
    text = " ".join(filtered_words)
    return text

def lemmatization(df, nlp, text_column="text", output_column="text"):
    """Lemmatizes the text in a specified column of a DataFrame and adds the results to a new column."""
    
    # Ensure the input column exists in the DataFrame
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' does not exist in the DataFrame.")
        
    # Apply SpaCy processing and lemmatization
    df[output_column] = df[text_column].apply(
        lambda text: " ".join([token.lemma_ for token in nlp(text) if not token.is_punct and not token.is_space]))
    
    return df

def POS_tagging(text, nlp):
    """Returns a list of (token, POS tag) tuples for the input text"""

    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

def preprocess_training_data():

    # Contruct a pandas DataFrame containing the training data
    directory = 'Classifier data'
    df = pd.DataFrame(columns=['article_id' , 'paragraph_id', 'text', 'group', 'publication_date'])
    article_nr = 1
    for folder in os.listdir(directory):
        for article in os.listdir(directory + '\\' + folder):
            file_path = os.path.join(directory, folder, article)
            text = extract_text_from_pdf(directory + '\\' + folder + '\\' + article)
            date = article.split(' ')[-1].split('.')[0] #Remove the article number and ".pdf" to obtain the publication date
            paragraphs = [para.strip() for para in text.split("\n \n") if para.strip()]
            para_nr = 1
            for para in paragraphs:
                df_temp = pd.DataFrame([[article_nr, para_nr, para, folder, date, file_path]], 
                                    columns=['article_id' , 'paragraph_id', 'text', 'group', 'publication_date', 'file_path'])
                df = pd.concat([df, df_temp])
                para_nr += 1
            article_nr += 1
    df.set_index(['article_id' , 'paragraph_id'], inplace=True)
    df['publication_date'] = pd.to_datetime(df['publication_date'], format='%d-%m-%Y')

    # Perpare for pre-processing
    model_name = "nl_core_news_sm"
    if not is_package(model_name):
        download(model_name)
    nlp = spacy.load('nl_core_news_sm')
    special_cases = {"[NEWLINE]": [{"ORTH": "[NEWLINE]"}]}
    nlp.tokenizer.add_special_case("[NEWLINE]", [{"ORTH": "[NEWLINE]"}])
    df['original_text'] = df['text'].copy()

    #Perform pre-processing and return the resulting DataFrame
    df['text'] = df['text'].apply(case_normalization)
    df['text'] = df['text'].apply(remove_punctuation)
    df['text'] = df['text'].apply(lambda text: remove_stopwords(text, nlp))
    df = lemmatization(df, nlp)
    df['pos_tags'] = df['text'].apply(lambda text: POS_tagging(text, nlp))
    print(df)
    
    return df