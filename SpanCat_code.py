# import fitz  # PyMuPDF
import os
import re
# import jsonlines
import json
import spacy
import random
from spacy.tokens import DocBin, Span, Doc
from spacy import displacy
import spacy_transformers
from spacy_transformers import TransformerModel
from spacy.training.example import Example
from transformers import AutoTokenizer, AutoModel, TFAutoModel
from spacy.util import minibatch
from spacy.vocab import Vocab

def convert_doccano_to_spacy(doccano_file, spacy_output_file, model="nl_core_news_lg"):
    """
    Converts a Doccano-labeled JSONL file into a spaCy-compatible .spacy file for Spancat training.
    
    Args:
    - doccano_file (str): Path to the input JSONL file from Doccano.
    - spacy_output_file (str): Path to save the output .spacy file.
    - model (str): The spaCy model to use for tokenization (default: "nl_core_news_lg").
    """

    # Load the specified model with NER disabled (to prevent conflicts with Spancat)
    nlp = spacy.load(model, exclude=["ner"])  
    doc_bin = DocBin()  # A container to store multiple spaCy Doc objects

    with open(doccano_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)  # Load each JSON object
            text = data["text"]  # Extract the text
            spans = data["label"]  # Extract the span annotations

            doc = nlp(text)  # Tokenize text using the selected model
            spancats = []  # Store spans as categorized spans

            for span in spans:
                start, end, label = span  # Extract start/end indices and label
                entity = doc.char_span(start, end, label=label, alignment_mode="contract")

                if entity:
                    spancats.append(entity)

            # Set spans in Spancat format
            doc.spans["sc"] = spancats  
            doc_bin.add(doc)  # Add the document to the batch

    # Save to disk in spaCy's binary format
    doc_bin.to_disk(spacy_output_file)

def SpanCat_data_prep():
    """
    Prepares the data for SpanCat training by extracting labeled texts from JSONL files and transforming them into Spacy files using nl_core_news_lg.
    """

    directory = 'SpanCat data'
    labels = ["Vastgoed", "Ondernemingen", "Arbeid", "Aansprakelijkheid & Letselschade"]
    pattern = re.compile(r"^(.*?)_training_spans_(\d+)$")

    max_indices = {
        label: (
            max(
                (
                    int(match.group(2))
                    for f in os.listdir(directory)
                    if (match := pattern.match(f)) and match.group(1) == label
                ),
                default=1
            ) # + 1 
        )
        for label in labels
    }
    print("\n Next indices for each label:\n", max_indices)

    for label, index in max_indices.items():
        print("label", label, "with index", index)
        convert_doccano_to_spacy(f"SpanCat data\\{label}\\{label}_training_spans_{index}.jsonl", 
                                 f"SpanCat data\\{label}\\{label}_training_spans_{index}.spacy", 
                                 model="nl_core_news_lg")  # Large model
        print("spacy file created for", label, "with index", index)

def train_SpanCat():
    """
    Trains and saves the SpanCat model using the prepared data.
    Returns a string indicating whether the training was successful.
    """

    directory = 'SpanCat data'
    labels = ["Vastgoed", "Ondernemingen", "Arbeid", "Aansprakelijkheid & Letselschade"]
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2]
    ngram_size = (2, 21)
    epochs = {"Vastgoed": 65, "Ondernemingen": 55, "Arbeid": 60, "Aansprakelijkheid & Letselschade": 70}

    for label in labels:
        #Find the most recent data for the currently selected label
        path = f"SpanCat data\\{label}"
        pattern = re.compile(rf"{re.escape(label)}_training_spans_(\d+)")
        files = os.listdir(path)
        matches = [(int(m.group(1)), f) for f in files if (m := pattern.fullmatch(os.path.splitext(f)[0]))]
        most_recent_file = max(matches, default=(None, None))[1]
        print("Most recent file for label", label, "is", most_recent_file)

        # Load the training data
        lang_model = spacy.load("nl_core_news_lg")  # Load the large Dutch language model
        doc_bin = DocBin().from_disk(f"{directory}\\{label}\\{most_recent_file}")
        docs = list(doc_bin.get_docs(lang_model.vocab))
        train_data = [Example(d, d) for d in docs]
        random.shuffle(train_data)
        print(f"label {label} training data loaded")
        print(train_data)

        # Create an empty SpanCat model and incrementally train it
        nlp = spacy.load("nl_core_news_lg")
        spancat = nlp.add_pipe("spancat", config={
            "spans_key": "sc",
            "suggester": {
                "@misc": "spacy.ngram_suggester.v1",
                "sizes": [i for i in range(ngram_size[0], ngram_size[1])]},
            "threshold": 0.0
        }, last=True)

        spancat.add_label(label)
        optimizer = nlp.initialize()

        for current_epoch in range(epochs[label]):
            local_random = random.Random(883821973 + current_epoch)
            local_random.shuffle(train_data)
            
            # Use minibatch training for efficiency
            losses = {}
            batches = minibatch(train_data, size=8)
            for batch in batches:
                nlp.update(batch, sgd=optimizer, losses=losses)

            if ((current_epoch + 1) % 5 == 0) or (current_epoch == epochs[label] - 1):
                print(f"Completed epoch {current_epoch+1}/{epochs[label]}")
        
        # Save model after training is complete
        match = re.search(r"_training_spans_(\d+)\.spacy", most_recent_file)
        index = match.group(1)  # Extracted number
        output_dir = f"SpanCat models\\{label}\\{label}_model_{index}"
        # output_dir = f"SpanCat models\\{label}\\{label}_model_{re.search(r'_training_spans_(\d+)\.spacy', most_recent_file).group(1)}"
        os.makedirs(output_dir, exist_ok=True)
        nlp.to_disk(output_dir)
        print(f"Model for label '{label}' saved to {output_dir}")
    
    return "Placeholder for training result"

def predict_spans(specialization, text):
    """
    Predicts spans in the given text using the trained SpanCat model for the specified specialization.
    
    Args:
    - specialization (str): The specialization for which to load the model.
    - text (str): The text to analyze.
    
    Returns:
    - list: A list of predicted spans with their labels.
    """
    
    directory = f"SpanCat models\\{specialization}"

    # Check if the model directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Model for specialization '{specialization}' not found in {directory}.")
    
    # Set max_index to the highest number in the model directory
    max_index = max((int(m.group(1)) for name in os.listdir(directory) if (m := re.search(r'_model_(\d+)$', name))), default=-1)

    if max_index == -1:
        raise ValueError(f"No model found for specialization '{specialization}' in {directory}.")

    # Load the appropriate model based on the specialization
    model_path = f"SpanCat models\\{specialization}\\{specialization}_model_{max_index}"
    nlp = spacy.load(model_path)
    doc = nlp(text)

    # for span in doc.spans.get("sc", []):
    #     if hasattr(span, "score") and span.score >= 0.1:
    #         print(f"Span: {span.text}, Label: {span.label_}, Score: {span.score:.2f}")
    




    
    # # Process the text
    # doc = nlp(text)
    
    # # Extract and return spans
    # spans = []
    # for span in doc.spans["sc"]:
    #     spans.append((span.text, span.label_))
    
    spans = []
    return spans