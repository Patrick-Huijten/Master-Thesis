import time
import fitz
import pandas as pd
import os
import string
import spacy
import torch
import logging
import re
import joblib
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from spacy.util import is_package
from spacy.cli import download
from transformers import AutoTokenizer, AutoModel, RobertaForSequenceClassification, RobertaTokenizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file using PyMuPDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: A single string containing the concatenated text from all pages in the PDF.
    """

    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def case_normalization(text: str) -> str:
    """
    Normalizes text by converting it to lowercase and replacing newlines with a placeholder.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text with lowercase characters and '[NEWLINE]' placeholders.
    """

    text = text.lower()
    text = text.replace('\n', ' [NEWLINE] ')
    while text != text.replace('  ', ' '):
        text = text.replace('  ', ' ')
    return text

def remove_punctuation(text: str) -> str:
    """
    Removes all punctuation from the input text while preserving '[NEWLINE]' tokens.

    Args:
        text (str): Input text.

    Returns:
        str: Text without punctuation.
    """
    
    text = text.translate(text.maketrans("", "", string.punctuation))
    text = text.replace("NEWLINE", "[NEWLINE]")
    return text

def remove_stopwords(text: str, nlp: spacy.lang) -> str:
    """
    Removes stopwords from the input text using a spaCy language model.

    Args:
        text (str): Input text.
        nlp (spacy.lang): Loaded spaCy language model.

    Returns:
        str: Text with stopwords removed.
    """
    
    doc = nlp(text)
    filtered_words = [token.text for token in doc if not token.is_stop]
    text = " ".join(filtered_words)
    return text

def lemmatization(df: pd.DataFrame, nlp: spacy.lang, text_column: str = "text", output_column: str = "text") -> pd.DataFrame:
    """
    Lemmatizes the text in a DataFrame column and stores the result in another column.

    Args:
        df (pd.DataFrame): Input DataFrame containing text data.
        nlp (spacy.lang): Loaded spaCy language model.
        text_column (str, optional): Column name of input text. Defaults to 'text'.
        output_column (str, optional): Column name to store lemmatized text. Defaults to 'text'.

    Returns:
        pd.DataFrame: DataFrame with an added or updated column of lemmatized text.

    Raises:
        ValueError: If the specified input column does not exist in the DataFrame.
    """
    
    # Ensure the input column exists in the DataFrame
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' does not exist in the DataFrame.")
        
    # Apply SpaCy processing and lemmatization
    df[output_column] = df[text_column].apply(
        lambda text: " ".join([token.lemma_ for token in nlp(text) if not token.is_punct and not token.is_space]))
    
    return df

def POS_tagging(text: str, nlp: spacy.lang) -> list:
    """
    Performs part-of-speech tagging on the input text using spaCy.

    Args:
        text (str): Input text.
        nlp (spacy.lang): Loaded spaCy language model.

    Returns:
        List of (token, POS tag) tuples.
    """

    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

def preprocess_training_data() -> pd.DataFrame:
    """
    Loads and preprocesses training data from PDF files into a DataFrame.

    The function reads PDF files from structured folders, extracts and splits text into paragraphs,
    and applies preprocessing steps including normalization, punctuation removal, stopword removal,
    lemmatization, and POS tagging.

    Returns:
        pd.DataFrame: Preprocessed DataFrame where each row represents a paragraph, 
        with columns for article/paragraph metadata and processed text.
    """

    directory = 'Classifier data'
    df = pd.DataFrame(columns=['article_id' , 'paragraph_id', 'text', 'group', 'publication_date'])
    article_nr = 1

    # Loop through all folders and articles in the directory and add them to the DataFrame
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

    return df

def get_word_embedding(text: str) -> np.ndarray:
    """
    Converts input text into a dense vector representation using the RobBERT transformer model.

    Args:
        text (str): Input text.

    Returns:
        np.ndarray: A 1D NumPy array representing the averaged word embeddings of the input.
    """

    model_name = "pdelobelle/robbert-v2-dutch-base"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Tokenize the input text
    with torch.no_grad():
        tokens = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        output = model(**tokens)
        embeddings = output.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

def train_classifier(df: pd.DataFrame) -> str:
    """
    Trains an SVM classifier on the preprocessed data using RobBERT embeddings and saves the model.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame with labeled text.

    Returns:
        str: Message shown to user, including pile path to the saved classifier model.
    """

    #There is no need to display warnings, so we disable them
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    #We select the parameter values which yielded the best performance. For more information, view the classification jupyter notebook
    C = 10
    kernel = "rbf"

    # Ensure the DataFrame has the necessary columns
    df["label"] = df["group"].astype("category").cat.codes 
    X_train = np.array([get_word_embedding(text) for text in df["text"]])
    y_train = df["label"].values

    # Train the SVM classifier
    classifier = SVC(kernel=kernel, C=C, probability=True)
    classifier.fit(X_train, y_train)
    
    #Create a folder for the model if such a folder does not yet exist
    model_dir = 'Classifier models'
    os.makedirs(model_dir, exist_ok=True)
    pattern = re.compile(r'^model_(\d+)$') # Regular expression to match file names with format model_X, where X is a positive int

    # Get all existing model files and extract their numbers
    existing_numbers = []
    for filename in os.listdir(model_dir):
        name, ext = os.path.splitext(filename)
        match = pattern.match(name)
        if match:
            existing_numbers.append(int(match.group(1)))
    
    # Save the model such that its associated number is the highest in the model folder
    next_number = max(existing_numbers, default=0) + 1
    new_model_name = f"model_{next_number}.joblib"
    save_path = os.path.join(model_dir, new_model_name)
    full_path = os.path.abspath(save_path)
    joblib.dump(classifier, save_path)
    
    return f"Classifier has been trained and saved to: {full_path}\n"

def summing_probabilities(df: pd.DataFrame) -> pd.Series:
    """
    Aggregates predicted class probabilities by summing them across all paragraphs of each article.

    Args:
        df (pd.DataFrame): DataFrame with paragraph-level predicted probabilities and article IDs.

    Returns:
        pd.Series: Series with article-level predictions based on summed normalized probabilities.
    """

    # For each article, sum the probabilities for each class across all paragraphs
    article_preds_prob = df.groupby('article_id')['pred_probabilities'].apply(lambda x: np.sum(np.array(x), axis=0))
    
    # Normalize the summed probabilities and select the class with the highest probability
    article_preds = article_preds_prob.apply(lambda x: np.argmax(x / np.sum(x)))
    return article_preds

def majority_voting(df: pd.DataFrame) -> pd.Series:
    """
    Predicts article-level class labels by applying majority voting on paragraph-level predictions.

    Args:
        df (pd.DataFrame): DataFrame containing paragraph-level predicted probabilities.

    Returns:
        pd.Series: Series with article-level majority-voted class predictions.
    """

    # For each paragraph, select the class with the highest probability
    df['pred_class'] = df['pred_probabilities'].apply(lambda x: np.argmax(x))
    
    # Now, for each article, apply majority voting
    article_preds_majority = df.groupby('article_id')['pred_class'].apply(lambda x: Counter(x).most_common(1)[0][0])
    return article_preds_majority

def combined_method(df: pd.DataFrame, X: float, Y: float) -> dict:
    """This function combines majority voting and summed probability normalization to predict the most likely class 
    for each article based on its paragraphs. The method works as follows:

    1. **Majority Voting**: For each paragraph, the class with the highest predicted probability is selected.
    2. **Tie-breaking with Summed Probabilities**: If there is a tie in the majority voting (i.e., multiple classes have equal counts), 
       the summed probabilities for each class across all paragraphs of an article are calculated and normalized. 
       The class with the highest summed probability is then selected.
    3. **Handling Uncertainty ("Unknown" Class)**: If the difference between the two highest summed probabilities is below a 
       specified threshold (X) or if the highest probability is below a minimum threshold (Y), the article is classified as "unknown".

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing the predicted class probabilities for each paragraph in an article, indexed by article_id and paragraph_id.
    X : float
        The threshold below which the difference between the two highest summed probabilities is considered too small to resolve a tie.
    Y : float
        The minimum summed probability for the highest class. If the highest probability is below this threshold, the article is classified as "unknown".

    Returns:
    --------
    article_preds_final : dict
        A dictionary mapping each article_id to its predicted class label, including "unknown" class predictions when applicable.
    """
    
    UNKNOWN_CLASS = -1
    df_reset = df.reset_index()

    # Step 1: Perform majority voting (predict most likely class for each paragraph)
    df_reset['pred_class'] = df_reset['pred_probabilities'].apply(lambda x: np.argmax(x))  # Most likely class per paragraph

    # Group by article_id and get top 2 predicted classes from majority voting
    df_reset['article_id'] = 1 #The code was initially written for multiple articles, so we need an id column, even if it is always the same id.
    article_preds_majority = df_reset.groupby('article_id')['pred_class'].apply(lambda x: Counter(x).most_common(2))

    # Step 2: Apply tie-breaking and fallback to summing probabilities
    article_preds_final = {}
    for article_id, top_preds in article_preds_majority.items():
        if len(top_preds) > 1:  # There's a tie
            # Get the top 2 predicted classes and their counts from majority voting
            class_1, count_1 = top_preds[0]
            class_2, count_2 = top_preds[1]
            
            # Get summed probabilities for both classes (sum across all paragraphs for each class)
            summed_probs = df_reset[df_reset['article_id'] == article_id].groupby('pred_class')['pred_probabilities'].apply(
                lambda x: np.sum(np.array(x), axis=0)
            )

            # Now, instead of summing across axis 1, we directly get the total sum of all probabilities across the article
            total_sum = np.sum([np.sum(probabilities) for probabilities in summed_probs])
            
            # Normalize the summed probabilities so that they sum to 1 across all classes
            summed_probs_normalized = summed_probs.apply(lambda x: x / total_sum)


            prob_1 = summed_probs_normalized.loc[class_1].sum()
            prob_2 = summed_probs_normalized.loc[class_2].sum()

            # Apply the threshold rules to decide if the article should be classified as unknown
            if abs(prob_1 - prob_2) < X or prob_1 < Y:
                article_preds_final[article_id] = UNKNOWN_CLASS
            else:
                # If there's no tie or the difference is significant, use the class with the higher probability
                article_preds_final[article_id] = class_1 if prob_1 > prob_2 else class_2
        else:
            # No tie, just use majority voting
            article_preds_final[article_id] = top_preds[0][0]

    return article_preds_final

def classify(df: pd.DataFrame, classifier: SVC) -> pd.DataFrame:
    """
    Classifies the text in the DataFrame using a pre-trained model and returns the predictions.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text to classify.
        model (sklearn.SVC): Pre-trained SVC model for classification.
        
    Returns:
        pd.DataFrame: DataFrame with predictions.
    """

    #There is no need to display warnings, so we disable them
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    # Prepare for pre-processing
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

    # Ensure the DataFrame has the necessary columns
    X = np.array([get_word_embedding(text) for text in df["text"]])
    y_pred_proba = classifier.predict_proba(X)
    df['pred_probabilities'] = list(y_pred_proba)

    # Get the final predictions using the combined method and constant values for X and Y
    X = 0.1
    Y = 0.3
    final_classification_dict = combined_method(df, X, Y)

    # Since combined_method was originally written for multiple articles, we need to convert the dictionary to a string of the first and only value
    class_mapping = {-1: "UNKNOWN — unable to determine a reliable class", 0: 'Bouw & Vastgoed', 1: 'Handel & Industrie', 2: 'Zakelijke Dienstverlening', 3: 'Zorg'}
    final_classification_pred = class_mapping[next(iter(final_classification_dict.values()))]

    return final_classification_pred, df[['pred_probabilities']]