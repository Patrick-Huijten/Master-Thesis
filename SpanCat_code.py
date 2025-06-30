import os
import re
import json
import spacy
import sys
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

def load_spacy_model(model_name="nl_core_news_md", exclude_ner=True) -> spacy.lang:
    """
    Loads a spaCy language model, either from a bundled path or from the installed models.

    Args:
        model_name (str): Name of the spaCy model to load (default is "nl_core_news_md").
        exclude_ner (bool): Whether to exclude the NER component from the model (default is True).

    Returns:
        spacy.lang: Loaded spaCy language model.
    """

    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    model_path = os.path.join(base_path, model_name)
    
    # Check subdirectory with version tag (PyInstaller often nests it this way)
    possible_subdirs = [d for d in os.listdir(model_path) if d.startswith(model_name)]
    if possible_subdirs:
        model_path = os.path.join(model_path, possible_subdirs[0])

    # Try loading the bundled model folder first
    if os.path.exists(model_path):
        if exclude_ner:
            return spacy.load(model_path, exclude=["ner"])
        else:
            return spacy.load(model_path)
    
    # Fallback to pip-installed version (e.g. dev environment)
    if exclude_ner:
        return spacy.load(model_name, exclude=["ner"])
    else:
        return spacy.load(model_name)

def convert_doccano_to_spacy(doccano_file: str, spacy_output_file: str, model: str = "nl_core_news_md") -> None:
    """
    Converts a Doccano-labeled JSONL file into a spaCy-compatible `.spacy` binary file for SpanCat training.
    
    Args:
        doccano_file (str): Path to the input JSONL file exported from Doccano.
        spacy_output_file (str): Path to save the converted `.spacy` binary file.
        model (str): The spaCy model used for tokenization (default: "nl_core_news_md").
        
    Returns:
        None
    """

    # Load the specified model with NER disabled (to prevent conflicts with Spancat)
    nlp = load_spacy_model(model, exclude_ner=True)
    doc_bin = DocBin()  # A container to store multiple spaCy Doc objects

    # Read the Doccano JSONL file line by line
    with open(doccano_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)  # Load each JSON object
            text = data["text"]  # Extract the text
            spans = data["label"]  # Extract the span annotations
            doc = nlp(text)  # Tokenize text using the selected model
            spancats = []  # Store spans as categorized spans

            # Iterate through the spans and create spaCy Span objects
            for span in spans:
                start, end, label = span  # Extract start/end indices and label
                entity = doc.char_span(start, end, label=label, alignment_mode="contract")

                # Check if the span is valid (not None)
                if entity:
                    spancats.append(entity)

            # Set spans in Spancat format
            doc.spans["sc"] = spancats  
            doc_bin.add(doc)  # Add the document to the batch

    # Save to disk in spaCy's binary format
    doc_bin.to_disk(spacy_output_file)

def SpanCat_data_prep() -> None:
    """
    Converts the most recent JSONL training file for each label into a `.spacy` binary file.

    Args:
        None
        
    Returns:
        None
    """

    directory = 'SpanCat data'
    labels = ["Vastgoed", "Ondernemingen", "Arbeid", "Aansprakelijkheid & Letselschade"]
    pattern = re.compile(r"^(.*?)_training_spans_(\d+)\.jsonl$")

    # For each label, find the latest JSONL training file and convert it to .spacy format
    for label in labels:
        label_dir = os.path.join(directory, label)
        if not os.path.exists(label_dir):
            continue
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

        # Convert the Doccano JSONL file to spaCy format
        convert_doccano_to_spacy(json_path, spacy_path, model="nl_core_news_md")
        print(f".spacy file created for {label} with index {index}")

def train_SpanCat() -> str:
    """
    Trains and saves a SpanCat model for each label using `.spacy` training files and the default embedding model.
    
    Args:
        None
        
    Returns:
        str: Message indicating training completion.
    """

    directory = 'SpanCat data'
    labels = ["Vastgoed", "Ondernemingen", "Arbeid", "Aansprakelijkheid & Letselschade"]
    ngram_size = (2, 21)
    epochs = {"Vastgoed": 55, "Ondernemingen": 45, "Arbeid": 60, "Aansprakelijkheid & Letselschade": 55}
    # epochs = {"Vastgoed": 15, "Ondernemingen": 15, "Arbeid": 15, "Aansprakelijkheid & Letselschade": 15} # replace with actual epochs once testing is done

    # For each label, find the most recent training file and train the SpanCat model
    for label in labels:
        path = f"SpanCat data\\{label}"
        pattern = re.compile(rf"{re.escape(label)}_training_spans_(\d+)")
        files = os.listdir(path)
        matches = [(int(m.group(1)), f) for f in files if (m := pattern.fullmatch(os.path.splitext(f)[0]))]
        most_recent_file = max(matches, default=(None, None))[1]
        print("Most recent file for label", label, "is", most_recent_file)

        # Load the language model & training data
        lang_model = load_spacy_model("nl_core_news_md", exclude_ner=False)
        doc_bin = DocBin().from_disk(f"{directory}\\{label}\\{most_recent_file}")
        docs = list(doc_bin.get_docs(lang_model.vocab))
        train_data = [Example(d, d) for d in docs]
        random.shuffle(train_data)
        print(f"label {label} training data loaded")
        # print(train_data)

        # Create an empty SpanCat model and incrementally train it
        nlp = load_spacy_model("nl_core_news_md", exclude_ner=True)
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
        
        # Save model after training is complete
        model_dir = f"SpanCat models\\{label}"
        os.makedirs(model_dir, exist_ok=True)
        existing_models = os.listdir(model_dir)
        model_pattern = re.compile(rf"{re.escape(label)}_model_(\d+)")
        existing_indices = [int(m.group(1)) for m in (model_pattern.fullmatch(os.path.splitext(f)[0]) for f in existing_models) if m]
        next_index = max(existing_indices, default=0) + 1

        output_dir = f"{model_dir}\\{label}_model_{next_index}"
        nlp.to_disk(output_dir)
        print(f"Model for label '{label}' saved to {output_dir}")
    
    return "Training Completed Successfully"

def train_SpanCat_incl_feature_vector() -> str:
    """
    Trains and saves a SpanCat model for each label using a custom tok2vec feature extractor for embedding.
    Uses token attributes and a maxout window encoder for richer span representations. 
    Trains for a fixed number of epochs and saves the model after training.
    
    Args:
        None
        
    Returns:
        str: Placeholder message indicating training completion.
    """

    print('')
    print("Training SpanCat with custom tok2vec feature extractor...")

    directory = 'SpanCat data'
    labels = ["Vastgoed", "Ondernemingen", "Arbeid", "Aansprakelijkheid & Letselschade"]
    ngram_size = (2, 21)
    # epochs = {"Vastgoed": 65, "Ondernemingen": 55, "Arbeid": 60, "Aansprakelijkheid & Letselschade": 70}
    epochs = {"Vastgoed": 55, "Ondernemingen": 45, "Arbeid": 60, "Aansprakelijkheid & Letselschade": 55}

    # For each label, find the most recent training file and train the SpanCat model
    for label in labels:
        path = f"SpanCat data\\{label}"
        pattern = re.compile(rf"{re.escape(label)}_training_spans_(\d+)")
        files = os.listdir(path)
        matches = [(int(m.group(1)), f) for f in files if (m := pattern.fullmatch(os.path.splitext(f)[0]))]
        most_recent_file = max(matches, default=(None, None))[1]
        print("Most recent file for label", label, "is", most_recent_file)

        # Load the training data
        lang_model = load_spacy_model("nl_core_news_md", exclude_ner=False)
        doc_bin = DocBin().from_disk(f"{directory}\\{label}\\{most_recent_file}")
        docs = list(doc_bin.get_docs(lang_model.vocab))
        train_data = [Example(d, d) for d in docs]
        random.shuffle(train_data)
        print(f"label {label} training data loaded")
        # print(train_data)

        # Create an empty SpanCat model and incrementally train it
        nlp = load_spacy_model("nl_core_news_md", exclude_ner=True)
        tok2vec_config = {
            "@architectures": "spacy.Tok2Vec.v2",
            "embed": {
                "@architectures": "spacy.MultiHashEmbed.v2",
                "width": 96,
                "attrs": ["ORTH", "LOWER", "PREFIX", "SUFFIX", "SHAPE"],
                "rows": [5000, 2500, 1000, 1000, 500]
            },
            "encode": {
                "@architectures": "spacy.MaxoutWindowEncoder.v2",
                "width": 96,
                "depth": 2,
                "window_size": 1,
                "maxout_pieces": 3
            }
        }

        spancat = nlp.add_pipe("spancat", config={
            "spans_key": "sc",
            "model": {
                "@architectures": "spacy.SpanCategorizer.v1",
                "tok2vec": tok2vec_config
            },
            "suggester": {
                "@misc": "spacy.ngram_suggester.v1",
                "sizes": [i for i in range(ngram_size[0], ngram_size[1])]
            },
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
        # Save model after training is complete
        model_dir = f"SpanCat models\\{label}"
        os.makedirs(model_dir, exist_ok=True)
        existing_models = os.listdir(model_dir)
        model_pattern = re.compile(rf"{re.escape(label)}_model_(\d+)")
        existing_indices = [int(m.group(1)) for m in (model_pattern.fullmatch(os.path.splitext(f)[0]) for f in existing_models) if m]
        next_index = max(existing_indices, default=0) + 1
        output_dir = f"{model_dir}\\{label}_model_{next_index}"
        nlp.to_disk(output_dir)
    
    return "Training Completed Successfully"

def predict_spans(specialization: str, text: str, threshold: float = 0.1) -> list:
    """
    Predicts categorized spans in the given text using the latest trained SpanCat model for the given specialization.
    
    Args:
        specialization (str): Name of the specialization (label) to load the corresponding model.
        text (str): Text input to analyze for span predictions.
        threshold (float): Minimum confidence score required to include a span prediction.
        
    Returns:
        list: List of dictionaries containing span text, character indices, label, and confidence score.
    """

    directory = f"SpanCat models\\{specialization}"

    # Check if the directory exists for the given specialization
    if not os.path.exists(directory):
        raise ValueError(f"Model for specialization '{specialization}' not found in {directory}.")

    # Find the latest model by checking for the highest index in the directory
    max_index = max((int(m.group(1)) for name in os.listdir(directory) if (m := re.search(r'_model_(\d+)$', name))), default=-1)

    if max_index == -1:
        raise ValueError(f"No model found for specialization '{specialization}' in {directory}.")

    # Load the latest model
    model_path = f"{directory}\\{specialization}_model_{max_index}"
    nlp = spacy.load(model_path)
    doc = nlp.make_doc(text)
    spancat = nlp.get_pipe("spancat")

    # Predict spans using the SpanCat model
    predictions = spancat.predict([doc])
    pred_spans = predictions[0].data.tolist()
    pred_scores = predictions[1].data.tolist()

    # Filter spans based on the threshold and prepare results
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

def editable_highlight_spans(text: str, spans: list) -> list:
    """
    Generates HTML components with clickable highlighted spans from a list of span predictions.
    This allows users to interact with the spans in a Dash application to add and/or remove spans.
    
    Args:
        text (str): The original input text.
        spans (list): A list of span dictionaries with 'start', 'end', 'label', and 'score' keys.
        
    Returns:
        list: A list of Dash `html.Span` components with styling and interactivity for visualization.
    """

    components = []
    current = 0

    # Fill the components list with all detected spans
    for i, span in enumerate(sorted(spans, key=lambda x: x['start'])):
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

    # Add any remaining text after the last span
    if current < len(text):
        components.append(html.Span(text[current:]))

    return components