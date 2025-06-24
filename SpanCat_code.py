# import fitz  # PyMuPDF
import os
import re
# import jsonlines
import json
import spacy
import random
from dash import html
from spacy.tokens import DocBin, Span, Doc
from spacy import displacy
import spacy_transformers
from spacy_transformers import TransformerModel
from spacy.training.example import Example
from transformers import AutoTokenizer, AutoModel, TFAutoModel
from spacy.util import minibatch
from spacy.vocab import Vocab

def convert_doccano_to_spacy(doccano_file, spacy_output_file, model="nl_core_news_md"):
    """
    Converts a Doccano-labeled JSONL file into a spaCy-compatible .spacy file for Spancat training.
    
    Args:
    - doccano_file (str): Path to the input JSONL file from Doccano.
    - spacy_output_file (str): Path to save the output .spacy file.
    - model (str): The spaCy model to use for tokenization (default: "nl_core_news_md").
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

# def SpanCat_data_prep():
#     """
#     Prepares the data for SpanCat training by extracting labeled texts from JSONL files and transforming them into Spacy files using nl_core_news_md.
#     """

#     directory = 'SpanCat data'
#     labels = ["Vastgoed", "Ondernemingen", "Arbeid", "Aansprakelijkheid & Letselschade"]
#     pattern = re.compile(r"^(.*?)_training_spans_(\d+)$")

#     max_indices = {
#         label: (
#             max(
#                 (
#                     int(match.group(2))
#                     for f in os.listdir(directory)
#                     if (match := pattern.match(f)) and match.group(1) == label
#                 ),
#                 default=1
#             ) # + 1 
#         )
#         for label in labels
#     }
#     print("\n Next indices for each label:\n", max_indices)

#     for label, index in max_indices.items():
#         print("label", label, "with index", index)
#         convert_doccano_to_spacy(f"SpanCat data\\{label}\\{label}_training_spans_{index}.jsonl", 
#                                  f"SpanCat data\\{label}\\{label}_training_spans_{index}.spacy", 
#                                  model="nl_core_news_md")  # Medium-sized model
#         print("spacy file created for", label, "with index", index)

def SpanCat_data_prep():
    """
    For each label, convert the most recent JSONL file to a .spacy file.
    """
    directory = 'SpanCat data'
    labels = ["Vastgoed", "Ondernemingen", "Arbeid", "Aansprakelijkheid & Letselschade"]
    pattern = re.compile(r"^(.*?)_training_spans_(\d+)\.jsonl$")

    for label in labels:
        label_dir = os.path.join(directory, label)
        if not os.path.exists(label_dir):
            continue

        # Find the latest JSONL training file
        matching_files = [
            f for f in os.listdir(label_dir)
            if (match := pattern.match(f)) and match.group(1) == label
        ]

        if not matching_files:
            print(f"No JSONL training file found for {label}")
            continue

        # Get max index
        latest_file = max(
            matching_files,
            key=lambda f: int(pattern.match(f).group(2))
        )
        index = int(pattern.match(latest_file).group(2))

        print(f"Converting for label {label}, using index {index}")
        json_path = os.path.join(label_dir, latest_file)
        spacy_path = os.path.join(label_dir, f"{label}_training_spans_{index}.spacy")

        convert_doccano_to_spacy(json_path, spacy_path, model="nl_core_news_md")
        print(f".spacy file created for {label} with index {index}")

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
    # epochs = {"Vastgoed": 15, "Ondernemingen": 15, "Arbeid": 15, "Aansprakelijkheid & Letselschade": 15} # replace with actual epochs once testing is done

    for label in labels:
        #Find the most recent data for the currently selected label
        path = f"SpanCat data\\{label}"
        pattern = re.compile(rf"{re.escape(label)}_training_spans_(\d+)")
        files = os.listdir(path)
        matches = [(int(m.group(1)), f) for f in files if (m := pattern.fullmatch(os.path.splitext(f)[0]))]
        most_recent_file = max(matches, default=(None, None))[1]
        print("Most recent file for label", label, "is", most_recent_file)

        # Load the training data
        lang_model = spacy.load("nl_core_news_md")  # Load the large Dutch language model
        doc_bin = DocBin().from_disk(f"{directory}\\{label}\\{most_recent_file}")
        docs = list(doc_bin.get_docs(lang_model.vocab))
        train_data = [Example(d, d) for d in docs]
        random.shuffle(train_data)
        print(f"label {label} training data loaded")
        print(train_data)

        # Create an empty SpanCat model and incrementally train it
        nlp = spacy.load("nl_core_news_md")
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
            print(f"starting epoch {current_epoch+1}/{epochs[label]} for label {label}")
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

# def predict_spans(specialization, text, threshold=0.1):
#     """
#     Predicts spans in the given text using the trained SpanCat model for the specified specialization.
    
#     Args:
#     - specialization (str): The specialization for which to load the model.
#     - text (str): The text to analyze.
#     - threshold (float): The minimum score to consider a span as valid.
    
#     Returns:
#     - list: A dictionary containing the following 3 lists relating to the input text: 'spans', 'labels', and 'scores'
#     """

#     directory = f"SpanCat models\\{specialization}"

#     # Check if the model directory exists
#     if not os.path.exists(directory):
#         raise ValueError(f"Model for specialization '{specialization}' not found in {directory}.")
    
#     # Set max_index to the highest number in the model directory
#     max_index = max((int(m.group(1)) for name in os.listdir(directory) if (m := re.search(r'_model_(\d+)$', name))), default=-1)

#     if max_index == -1:
#         raise ValueError(f"No model found for specialization '{specialization}' in {directory}.")

#     # Load the appropriate model based on the specialization
#     model_path = f"SpanCat models\\{specialization}\\{specialization}_model_{max_index}"
#     nlp = spacy.load(model_path)

#     # Prepare and tokenize the document
#     doc = nlp.make_doc(text)
#     spancat = nlp.get_pipe("spancat")

#     # Predict spans once
#     predictions = spancat.predict([doc])
#     pred_spans = predictions[0].data.tolist()  # span indices
#     pred_scores = predictions[1].data.tolist()  # confidence scores

#     # Filter and return spans based on the specified threshold
#     results = []
#     for (start_token, end_token), scores in zip(pred_spans, pred_scores):
#         span = doc[start_token:end_token]  # this gives a Span object (token-based)

#         for label, score in zip(spancat.labels, scores):
#             if score >= threshold:
#                 results.append({
#                     "text": span.text,
#                     "start": span.start_char,
#                     "end": span.end_char,
#                     "label": label,
#                     "score": score
#                 })

#     # results = []
#     # for (start, end), scores in zip(pred_spans, pred_scores):
#     #     for label, score in zip(spancat.labels, scores):
#     #         if score >= threshold:
#     #             span = doc[start:end]
#     #             results.append({
#     #                 "text": span.text,
#     #                 "start": start,
#     #                 "end": end,
#     #                 "label": label,
#     #                 "score": score
#     #             })

#     print(results, len(results))
#     # print('')
#     # print(results['scores'][:10],results['labels'][:10],results['spans'][:10])

#     return results

def predict_spans(specialization, text, threshold=0.1):
    """
    Predicts spans in the given text using the trained SpanCat model for the specified specialization.

    Args:
    - specialization (str): The specialization for which to load the model.
    - text (str): The text to analyze.
    - threshold (float): The minimum score to consider a span as valid.

    Returns:
    - list: A list of span dictionaries with keys 'text', 'start', 'end', 'label', and 'score'.
    """

    directory = f"SpanCat models\\{specialization}"

    if not os.path.exists(directory):
        raise ValueError(f"Model for specialization '{specialization}' not found in {directory}.")

    max_index = max((int(m.group(1)) for name in os.listdir(directory) if (m := re.search(r'_model_(\d+)$', name))), default=-1)

    if max_index == -1:
        raise ValueError(f"No model found for specialization '{specialization}' in {directory}.")

    model_path = f"{directory}\\{specialization}_model_{max_index}"
    nlp = spacy.load(model_path)

    doc = nlp.make_doc(text)
    spancat = nlp.get_pipe("spancat")

    predictions = spancat.predict([doc])
    pred_spans = predictions[0].data.tolist()
    pred_scores = predictions[1].data.tolist()

    results = []
    for (start_token, end_token), scores in zip(pred_spans, pred_scores):
        span = doc[start_token:end_token]
        for label, score in zip(spancat.labels, scores):
            if score >= threshold:
                results.append({
                    "text": span.text,
                    "start": span.start_char,
                    "end": span.end_char,
                    "label": label,
                    "score": score
                })

    return results

def editable_highlight_spans(text, spans):
    components = []
    current = 0

    for i, span in enumerate(sorted(spans, key=lambda x: x['start'])):
        # Add non-highlighted text before span
        if current < span['start']:
            components.append(html.Span(text[current:span['start']]))

        # Add span with onclick and index
        span_text = text[span['start']:span['end']]
        components.append(
            html.Span(
                span_text,
                id={'type': 'highlighted-span', 'index': i},
                n_clicks=0,
                style={
                    'backgroundColor': "#127bf3",
                    'padding': '2px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'margin': '1px'
                },
                title=f"Label: {span['label']}, Score: {span['score']:.2f}"
            )
        )
        current = span['end']

    if current < len(text):
        components.append(html.Span(text[current:]))

    return components

    # for span in doc.spans.get("sc", []):
    #     if hasattr(span, "score") and span.score >= 0.1:
    #         print(f"Span: {span.text}, Label: {span.label_}, Score: {span.score:.2f}")
    


    
    # # Process the text
    # doc = nlp(text)
    
    # # Extract and return spans
    # spans = []
    # for span in doc.spans["sc"]:
    #     spans.append((span.text, span.label_))
    
    # spans = []
    # return spans

# def predict_spans(specialization, text):
#     """
#     Predicts spans in the given text using the trained SpanCat model for the specified specialization.
    
#     Args:
#     - specialization (str): The specialization for which to load the model.
#     - text (str): The text to analyze.
    
#     Returns:
#     - list: A list of predicted spans with their labels.
#     """
    
#     directory = f"SpanCat models\\{specialization}"

#     # Check if the model directory exists
#     if not os.path.exists(directory):
#         raise ValueError(f"Model for specialization '{specialization}' not found in {directory}.")
    
#     # Set max_index to the highest number in the model directory
#     max_index = max((int(m.group(1)) for name in os.listdir(directory) if (m := re.search(r'_model_(\d+)$', name))), default=-1)

#     if max_index == -1:
#         raise ValueError(f"No model found for specialization '{specialization}' in {directory}.")

#     # Load the appropriate model based on the specialization
#     model_path = f"SpanCat models\\{specialization}\\{specialization}_model_{max_index}"
#     nlp = spacy.load(model_path)
#     doc = nlp(text)

#     # for span in doc.spans.get("sc", []):
#     #     if hasattr(span, "score") and span.score >= 0.1:
#     #         print(f"Span: {span.text}, Label: {span.label_}, Score: {span.score:.2f}")
    




    
#     # # Process the text
#     # doc = nlp(text)
    
#     # # Extract and return spans
#     # spans = []
#     # for span in doc.spans["sc"]:
#     #     spans.append((span.text, span.label_))
    
#     spans = []
#     return spans