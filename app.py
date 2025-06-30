print("Loading, please wait...")

import base64
import io
import fitz  # PyMuPDF
import dash
from datetime import datetime
import joblib
import json
import pandas as pd
import plotly.graph_objects as go
import re
import textwrap
import tempfile
import os
import webbrowser
import matplotlib.pyplot as plt
from dash import html, dcc, Input, Output, State, Dash, ctx, clientside_callback, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from flask import send_file, request
from reportlab.lib.colors import black, white, HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.units import inch
from threading import Timer
from waitress import serve
import warnings
import spacy

# External stylesheet
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MannaertsAppels Dashboard"
app.server.max_content_length = 1024 * 1024 * 1000  # 1GB limit

# Import training logic
from Sector_Classification import train_classifier, preprocess_training_data, classify
from SpanCat_code import SpanCat_data_prep, train_SpanCat, predict_spans, editable_highlight_spans, train_SpanCat_incl_feature_vector

# PDF text extraction
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts and returns all text content from a PDF file in bytes format.

    Args:
        pdf_bytes (bytes): Byte stream of the PDF file.

    Returns:
        str: The combined text from all pages of the PDF.
    """

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

# App layout
app.layout = dbc.Container([

    # Save the spans found by SpanCat for reference
    dcc.Store(id="current-spans", data=[]),

    # Store for the current text content
    dcc.Store(id="current-text", data=""),

    # Store for the selected text range for manually adding spans
    dcc.Store(id="selected-text-store", data={"start": None, "end": None}),

    # Store the user-selected decision threshold
    dcc.Store(id="threshold-store", data=0.1),

    # Download component for exporting annotated pdf files
    dcc.Download(id="download-pdf"),

    # Title
    dbc.Row([ 
        dbc.Col(
            dbc.Card([ 
                dbc.CardHeader( 
                    html.Div([ 
                        "Mannaerts",
                        html.Span("Appels", style={"color": "#ea1a8d", "fontWeight": "bold"})
                    ], className="text-center fs-3 fw-bold")
                )
            ], className="mb-4"),
            width=12
        )
    ]), 

    # Main content: Text area + Buttons + Placeholder
    dbc.Row([ 
        # Left column: Text area
        dbc.Col([ 
            dbc.Card([ 
                dbc.CardBody([ 

                    # Text area for displaying PDF text
                    html.Div(
                        "Upload a PDF file using the button on the right. The contents will appear here.",
                        id='highlighted-text-display',
                        style={'height': '550px', 'overflowY': 'scroll', 'whiteSpace': 'pre-wrap', 'fontSize': '20px'}
                    ),

                    # Hidden div to store current spans
                    html.Div([ 
                        "Font Size", 
                        dcc.Slider( 
                            id="font-size-slider",
                            min=10, 
                            max=40, 
                            step=1, 
                            value=20,  # Default value
                            marks={i: str(i) for i in range(10, 41, 5)},  # Mark every 5 units
                        ),

                        "SpanCat Threshold",
                        dcc.Slider(
                            id="threshold-slider",
                            min=0.0,
                            max=1.0,
                            step=0.001,
                            value=0.1,  # Default threshold
                            marks={i: f"{str(round(i, 2))}" for i in [i * 0.01 for i in range(0, 101, 10)]},
                            tooltip={"always_visible": False, "placement": "bottom"},
                        ),

                        # Buttons for adding spans and saving data
                        dbc.Button("Add Selected Span", id="add-span-btn", color="success", className="mt-2"),
                        dbc.Button("Add Article to SpanCat Training Data", id="save-spancat-btn", color="success", className="mt-2", style={"marginLeft": "10px"}),
                        dbc.Button("Export Annotated PDF", id="export-pdf-btn", color="primary", className="mt-2", style={"marginLeft": "10px"})

                    ], style={"marginTop": "20px"})
                ]) 
            ]) 
        ], width=5), 

        dbc.Col([ 
            dbc.Row([ 
                dbc.Col([ 
                    # Upload button for PDF files:
                    dcc.Upload( 
                        id='upload-data', 
                        children=dbc.Button("Select PDF File", color="primary", className="w-100", size="sm"), 
                        multiple=False
                    )
                ], width=3),

                # Buttons for loading models and retraining
                dbc.Col([ 
                    dcc.Upload(
                        id='upload-model',
                        children=dbc.Button("Load Classifier Model", color="info", className="w-100", size="sm"),
                        multiple=False,
                        accept=".joblib"
                    )
                ], width=3),
                dbc.Col([ 
                    dbc.Button("Load SpanCat Model", id="load-spancat-button", color="info", className="w-100", size="sm") 
                ], width=3),
                dbc.Col([ 
                    dbc.Button("Re-train Models", id="retrain-button", color="danger", className="w-100", size="sm") 
                ], width=3),
            ], className="mb-3"), 

            dbc.Row([
                dbc.Col(
                    id="sector_pred",
                    width=8
                ),

                # Dropdown + button to save the uploaded article to a specific class' training dataset
                dbc.Col([
                    dcc.Dropdown(
                        id="true-class-dropdown",
                        options=[
                            {"label": "Bouw & Vastgoed", "value": "Bouw & Vastgoed"},
                            {"label": "Handel & Industrie", "value": "Handel & Industrie"},
                            {"label": "Zakelijke Dienstverlening", "value": "Zakelijke Dienstverlening"},
                            {"label": "Zorg", "value": "Zorg"}
                        ],
                        placeholder="Select true class",
                        style={"marginBottom": "10px"}
                    ),
                    dbc.Button(
                        "Add Article to Classifier Training Data",
                        id="save-pdf-sector",
                        color="success",
                        className="w-100",
                        size="sm"
                    )
                ], width=4)
            ], className="mb-4", align="start"),

            # Visualization area for classification results per paragraph
            dcc.Graph(figure={}, id="classification_plot", style={"height": "400px"}),

            #Custom warning if newly added span overlaps with existing span(s)
            dbc.Toast(
                id="overlap-warning-toast",
                header="Warning",
                is_open=False,
                dismissable=True,
                icon="danger",
                duration=4000,
                children="Selected span overlaps with an existing one and cannot be added.",
                style={"position": "fixed", "top": 10, "right": 10, "width": 350}
            )
        ], width=7)
    ], className="mb-4"),

    # Modal for retraining confirmation
    dbc.Modal([ 
        dbc.ModalHeader("Confirm Retraining"), 
        dbc.ModalBody("This may take a while. Are you sure you want to proceed?"), 
        dbc.ModalFooter([ 
            dbc.Button("Cancel", id="cancel-retrain", color="secondary", className="me-2"), 
            dbc.Button("Confirm", id="confirm-retrain", color="danger") 
        ]) 
    ], id="retrain-modal", is_open=False),

    # Modal for selecting class-specific SpanCat model
    dbc.Modal([
    dbc.ModalBody([
        html.Div(
            "Please select your desired specialization. " \
            "The latest SpanCat model for your selection will be loaded",
            style={"marginBottom": "15px"}
        ),
        dbc.RadioItems(
            id="SpanCat-specialization-radio",
            options=[
                {"label": "Vastgoed", "value": "Vastgoed"},
                {"label": "Ondernemingen", "value": "Ondernemingen"},
                {"label": "Arbeid", "value": "Arbeid"},
                {"label": "Aansprakelijkheid & Letselschade", "value": "Aansprakelijkheid & Letselschade"}
            ],
            value="Vastgoed",
            inline=False
        ),
    ]),
    dbc.ModalFooter([
        dbc.Button("Cancel", id="cancel-model-select", color="secondary", className="me-2"),
        dbc.Button("Confirm", id="confirm-model-select", color="primary"),
    ]),
], id="Spancat-select-modal", is_open=False),

    # Loading spinner and retrain status to visually indicate retraining progress
    dbc.Row([ 
        dbc.Col([ 
            dcc.Loading( 
                id="loading-spinner", 
                type="circle", 
                color="#00ff00", 
                children=html.Div(id="retrain-status", className="mt-3")
            ) 
        ]) 
    ])
], fluid=True)

#____________________________________________________________________________________________________
# Callbacks

app.clientside_callback(
    """
    function(n_clicks) {
        const sel = window.getSelection();
        if (!sel || sel.rangeCount === 0) return window.dash_clientside.no_update;

        const range = sel.getRangeAt(0);
        const container = document.getElementById('highlighted-text-display');
        if (!container || !container.contains(range.commonAncestorContainer)) return window.dash_clientside.no_update;

        const selected = range.toString();

        return {
            text: selected
        };
    }
    """,
    Output("selected-text-store", "data"),
    Input("add-span-btn", "n_clicks"),
    prevent_initial_call=True
)

# Callback: Toggle SpanCat modal open/close
@app.callback(
    Output("Spancat-select-modal", "is_open"),
    [Input("load-spancat-button", "n_clicks"),
     Input("confirm-model-select", "n_clicks"),
     Input("cancel-model-select", "n_clicks")],
)
def toggle_SpanCat_modal(upload_clicks, confirm_clicks, cancel_clicks):
    """
    Toggles the modal for SpanCat model selection based on user interaction.

    Args:
        upload_clicks (int): Clicks on the "Load SpanCat Model" button.
        confirm_clicks (int): Clicks on the "Confirm" button.
        cancel_clicks (int): Clicks on the "Cancel" button.

    Returns:
        bool: Whether the modal should be open.
    """

    triggered = ctx.triggered_id
    if triggered == "load-spancat-button":
        return True
    elif triggered in ["confirm-model-select", "cancel-model-select"]:
        return False
    return dash.no_update

# Callback: Toggle retrain modal open/close
@app.callback(
    Output("retrain-modal", "is_open"),
    [Input("retrain-button", "n_clicks"),
     Input("confirm-retrain", "n_clicks"),
     Input("cancel-retrain", "n_clicks")],
    [State("retrain-modal", "is_open")]
)
def toggle_retrain_modal(retrain_clicks, confirm_clicks, cancel_clicks, is_open):
    """
    Toggles the modal that confirms retraining based on user interaction.

    Args:
        retrain_clicks (int): Clicks on the "Retrain Models" button.
        confirm_clicks (int): Clicks on the "Confirm" button.
        cancel_clicks (int): Clicks on the "Cancel" button.
        is_open (bool): Current state of the modal.

    Returns:
        bool: Updated modal open state.
    """

    triggered = ctx.triggered_id
    if triggered == "retrain-button":
        return True
    elif triggered in ["confirm-retrain", "cancel-retrain"]:
        return False
    return is_open

# Callback: Train model
@app.callback(
    Output("retrain-status", "children"),
    Input("confirm-retrain", "n_clicks"),
    prevent_initial_call=True
)   
def retrain_model(n_clicks):
    """
    Initiates retraining for both the classifier and SpanCat models.

    Args:
        n_clicks (int): Clicks on the "Confirm" retraining button.

    Returns:
        html.Span: Status message indicating success or failure.
    """

    if not n_clicks:
        raise PreventUpdate
    try:
        # Pre-process training data and train the classifier
        df_train = preprocess_training_data()
        status = train_classifier(df_train)

        # Prepare data for SpanCat and train the model
        SpanCat_data_prep()
        # status += train_SpanCat()
        status += train_SpanCat_incl_feature_vector() # If you don't want to include feature vectors (experimental), comment this line and uncomment the previous line

        print(f"Status={status}")

        return html.Span(status, style={"color": "limegreen", "fontWeight": "bold"})
    except Exception as e:
        return html.Span(f"Training failed: {str(e)}", style={"color": "red", "fontWeight": "bold"})

# Callback: Update font size of the text area
@app.callback(
    # Output("pdf-text-output", "style"),
    Output("highlighted-text-display", "style"),
    Input("font-size-slider", "value")
)
def update_font_size(font_size):
    """
    Updates the font size for the displayed slider.

    Args:
        font_size (int): Selected font size from the slider.

    Returns:
        dict: Updated CSS style dict with the new font size.
    """
            
    return {
        'width': '100%',
        'height': '550px',
        'overflowY': 'scroll',
        'whiteSpace': 'pre-wrap',
        'fontSize': f'{font_size}px'
        }

@app.callback(
    Output("highlighted-text-display", "children"),
    Output("current-spans", "data"),
    Output("current-text", "data"),
    Output("sector_pred", "children"),
    Output("classification_plot", "figure"),
    Output("overlap-warning-toast", "is_open"),
    Output("threshold-store", "data"),
    Input("upload-model", "contents"),
    Input("upload-data", "contents"),
    Input("confirm-model-select", "n_clicks"),
    Input({'type': 'highlighted-span', 'index': dash.ALL}, 'n_clicks'),
    Input("selected-text-store", "data"),
    Input("threshold-slider", "value"),
    State("SpanCat-specialization-radio", "value"),
    State("current-spans", "data"),
    State("current-text", "data"),
    State("threshold-store", "data"),
    prevent_initial_call=True
)
def unified_handler(classifier_contents, pdf_contents, confirm_spancat_clicks, span_clicks, selection, threshold_slider_val, 
                    specialization, spans, text, stored_threshold):
    """
    Handles multiple user actions including PDF upload, model upload, classification, span addition/removal, 
    threshold changes, and SpanCat predictions.

    This is the core logic that reacts to updates from the UI and orchestrates data preprocessing, 
    classification predictions, span predictions, visual rendering, and warning triggers.

    Args:
        classifier_contents (str): Base64-encoded content of the uploaded classifier model.
        pdf_contents (str): Base64-encoded content of the uploaded PDF file.
        confirm_spancat_clicks (int): Clicks on the SpanCat "Confirm" button.
        span_clicks (list): List of clicks on individual highlighted spans.
        selection (dict): Dictionary with the selected text from the UI.
        threshold_slider_val (float): Current threshold value from the slider.
        specialization (str): Selected SpanCat specialization (e.g., "Vastgoed").
        spans (list): Current list of highlighted spans.
        text (str): Current text content from the uploaded PDF.
        stored_threshold (float): Previously stored threshold value.

    Returns:
        list: Updated Dash HTML components displaying highlighted spans or raw text.
        list: Updated list of span dictionaries.
        str: Current text content.
        html.Div or None: Classification result component (or None if no prediction).
        plotly.graph_objects.Figure: Bar chart of predicted class probabilities.
        bool: Whether the overlap warning toast should be shown.
        float: Updated SpanCat threshold value to store.
    """

    # Initialize variables
    fig = go.Figure()
    prediction_display = None
    triggered_id = ctx.triggered_id

    # Case 1: Remove a span
    if isinstance(triggered_id, dict) and triggered_id.get("type") == "highlighted-span":
        index_to_remove = triggered_id.get("index")
        new_spans = spans[:index_to_remove] + spans[index_to_remove + 1:]
        new_components = editable_highlight_spans(text, new_spans)
        return new_components, new_spans, text, dash.no_update, dash.no_update, False, threshold_slider_val

    # Case 2: Add a new span
    if triggered_id == "selected-text-store":
        if not selection or selection.get("text") is None:
            raise PreventUpdate

        selected_text = selection["text"]
        start = text.find(selected_text)
        end = start + len(selected_text)

        if start == -1 or start == end or end > len(text):
            raise PreventUpdate

        for span in spans:
            if not (end <= span["start"] or start >= span["end"]):
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, True, threshold_slider_val

        new_span = {
            "text": selected_text,
            "start": start,
            "end": end,
            "label": specialization,
            "score": 1.0
        }

        updated_spans = spans + [new_span]
        new_components = editable_highlight_spans(text, updated_spans)
        return new_components, updated_spans, text, dash.no_update, dash.no_update, False, threshold_slider_val

    # Case 3: Load new text from PDF
    if pdf_contents:
        content_type, content_string = pdf_contents.split(',')
        decoded_pdf = base64.b64decode(content_string)
        text = extract_text_from_pdf(decoded_pdf)

    # Case 4: Load model and classify
    if pdf_contents and classifier_contents:
        def decode_model(contents):
            _, content_string = contents.split(',')
            return io.BytesIO(base64.b64decode(content_string))

        try:
            model = joblib.load(decode_model(classifier_contents))
        except Exception as e:
            print("Error loading model:", e)
            return text, [], text, None, fig, False, threshold_slider_val

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        df_new = pd.DataFrame({"text": paragraphs})
        result_string, df_class_probas = classify(df_new, model)

        class_mapping = {
            0: 'Bouw & Vastgoed',
            1: 'Handel & Industrie',
            2: 'Zakelijke Dienstverlening',
            3: 'Zorg'
        }
        colors = {
            'UNKNOWN — unable to determine a reliable class': '#888888',
            'Bouw & Vastgoed': '#009E73',
            'Handel & Industrie': '#0072B2',
            'Zakelijke Dienstverlening': '#CC79A7',
            'Zorg': '#D55E00'
        }

        proba_values = df_class_probas['pred_probabilities'].tolist()
        df_probs = pd.DataFrame(proba_values).rename(columns=class_mapping)
        paragraph_labels = [f"P{i + 1}" for i in range(len(df_probs))]

        for class_name in df_probs.columns:
            fig.add_trace(go.Bar(
                x=paragraph_labels,
                y=df_probs[class_name],
                name=class_name,
                marker_color=colors.get(class_name, '#888'),
                hovertemplate=f"%{{x}}<br>{class_name}: %{{y:.2%}}<extra></extra>"
            ))

        fig.update_layout(
            title="Sector Probabilities Per Paragraph",
            barmode='stack',
            xaxis_title='Paragraph',
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            legend_title='Class',
            template='plotly_dark'
        )

        color = colors.get(result_string, '#ffffff')
        prediction_display = html.Div([
            html.Span("Prediction: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.Span(result_string, style={"color": color, "fontWeight": "bold", "fontSize": "24px"})
        ])

    # Case 5: Predict spans or threshold changed
    spans = []
    highlighted = None
    if pdf_contents and (confirm_spancat_clicks or triggered_id == "threshold-slider"):
        spans = predict_spans(specialization, text, threshold=threshold_slider_val)
        highlighted = editable_highlight_spans(text, spans)

    if text and not highlighted:
        highlighted = text

    return highlighted, spans, text, prediction_display, fig, False, threshold_slider_val

@app.callback(
    Output("retrain-status", "children", allow_duplicate=True),
    Input("save-pdf-sector", "n_clicks"),
    State("true-class-dropdown", "value"),
    State("upload-data", "contents"),
    prevent_initial_call=True
)
def save_pdf_to_class_folder(n_clicks, selected_class, pdf_contents):
    """
    Saves the uploaded PDF to the classifier training folder with a unique filename.

    Args:
        n_clicks (int): Clicks on the "Add Article to Classifier Training Data" button.
        selected_class (str): Selected true class/category.
        pdf_contents (str): Base64 encoded content of the uploaded PDF.

    Returns:
        dbc.Alert: Status message about the save operation.
    """

    if not pdf_contents or not selected_class:
        raise PreventUpdate

    try:
        # Make sure the base directory exists
        base_dir = "Classifier data"
        class_dir = os.path.join(base_dir, selected_class)
        os.makedirs(class_dir, exist_ok=True)

        # Find highest existing ID so far and increment it
        existing_files = os.listdir(class_dir)
        id_pattern = re.compile(r'^(\d+),\s*\d{2}-\d{2}-\d{4}\.pdf$')
        existing_ids = []
        for fname in existing_files:
            match = id_pattern.match(fname)
            if match:
                existing_ids.append(int(match.group(1)))
        next_id = max(existing_ids) + 1 if existing_ids else 1

        # Use current date as proxy for publication date
        today_str = datetime.now().strftime("%d-%m-%Y")

        # Construct filename
        new_filename = f"{next_id}, {today_str}.pdf"
        file_path = os.path.join(class_dir, new_filename)

        # Decode and save
        _, content_string = pdf_contents.split(',')
        pdf_data = base64.b64decode(content_string)

        with open(file_path, "wb") as f:
            f.write(pdf_data)

        return dbc.Alert(f"Saved file as '{new_filename}' in folder '{selected_class}'.", color="success", dismissable=True)

    except Exception as e:
        return dbc.Alert(f"Error saving PDF: {e}", color="danger", dismissable=True)

@app.callback(
    Output("retrain-status", "children", allow_duplicate=True),
    Input("save-spancat-btn", "n_clicks"),
    State("SpanCat-specialization-radio", "value"),
    State("current-text", "data"),
    State("current-spans", "data"),
    prevent_initial_call=True
)
def save_spancat_training_data(n_clicks, specialization, text, spans):
    """
    Adds the current text and its associated spans to copy of the SpanCat training data for the selected specialization.

    Args:
        n_clicks (int): Clicks on the "Add Article to SpanCat Training Data" button.
        specialization (str): Selected SpanCat specialization.
        text (str): Document text.
        spans (list): Annotated spans.

    Returns:
        dbc.Alert: Status message indicating result of the save operation.
    """

    if not text or not spans or not specialization:
        raise PreventUpdate

    try:
        # Directory setup
        base_dir = "SpanCat data"
        class_dir = os.path.join(base_dir, specialization)
        os.makedirs(class_dir, exist_ok=True)

        # Find the highest existing index in filenames
        pattern = re.compile(rf"{re.escape(specialization)}_training_spans_(\d+)\.jsonl")
        existing_files = [f for f in os.listdir(class_dir) if pattern.fullmatch(f)]
        existing_indices = [int(pattern.fullmatch(f).group(1)) for f in existing_files]
        next_index = max(existing_indices) + 1 if existing_indices else 1

        # Load previous data from most recent file, if it exists
        combined_data = []
        if existing_files:
            latest_file = os.path.join(class_dir, f"{specialization}_training_spans_{max(existing_indices)}.jsonl")
            with open(latest_file, "r", encoding="utf-8") as f:
                combined_data = [json.loads(line) for line in f if line.strip()]

        # Add the new example
        new_entry = {
            "text": text,
            "label": [[span["start"], span["end"], span["label"]] for span in spans]
        }
        combined_data.append(new_entry)

        # Save to a new file with next index
        new_filename = f"{specialization}_training_spans_{next_index}.jsonl"
        new_path = os.path.join(class_dir, new_filename)
        with open(new_path, "w", encoding="utf-8") as f:
            for entry in combined_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return dbc.Alert(f"Saved new training file: {new_filename}", color="success", dismissable=True)

    except Exception as e:
        return dbc.Alert(f"Failed to save SpanCat data: {e}", color="danger", dismissable=True)

@app.callback(
    Output("download-pdf", "data"),
    Input("export-pdf-btn", "n_clicks"),
    State("current-text", "data"),
    State("current-spans", "data"),
    prevent_initial_call=True
)
def download_pdf_direct(n_clicks, text, spans):
    """
    Generates a downloadable PDF with highlighted spans directly from the UI.

    Args:
        n_clicks (int): Clicks on the "Export Annotated PDF" button.
        text (str): Original document text.
        spans (list): Highlighted spans.

    Returns:
        dcc.send_bytes: A downloadable annotated PDF file.
    """

    if not text or not spans:
        return no_update

    # Generate pdf in memory
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    x_margin = inch
    y = height - inch
    font_size = 11
    line_height = 16
    max_line_width = width - 2 * x_margin
    font_name = "Helvetica"
    c.setFont(font_name, font_size)

    # Mark spans by index
    span_lookup = {i: span for span in spans for i in range(span["start"], span["end"])}

    def draw_line_with_highlights(line_text, global_char_offset):
        """
        Draws a line of text to the PDF canvas, highlighting segments that fall within annotated spans.

        Args:
            line_text (str): The line of text to render.
            global_char_offset (int): The character offset of the line within the entire document.

        Returns:
            None
        """

        # Set initial x position and prepare to iterate through the line
        nonlocal y
        x = x_margin
        i = 0

        # Iterate through the line text, checking each character against the span lookup
        while i < len(line_text):
            # Determine if the current character is highlighted and find the extent of the highlight
            is_highlighted = (global_char_offset + i) in span_lookup
            j = i
            while j < len(line_text) and ((global_char_offset + j) in span_lookup) == is_highlighted:
                j += 1
            chunk = line_text[i:j]
            width_chunk = stringWidth(chunk, font_name, font_size)
            if is_highlighted:
                c.setFillColor(HexColor("#127bf3"))
                c.rect(x - 1, y - 3, width_chunk + 2, line_height + 2, fill=True, stroke=False)
                c.setFillColor(white)
            else:
                c.setFillColor(black)
            c.drawString(x, y, chunk)
            x += width_chunk
            i = j
        y -= line_height
        if y < inch:
            c.showPage()
            c.setFont(font_name, font_size)
            y = height - inch

    char_index = 0
    lines = text.splitlines(keepends=True)  # Keeps newlines

    for line in lines:
        line_content = line.rstrip('\r\n')
        wrapped_lines = textwrap.wrap(line_content, width=100) or ['']
        offset_in_line = 0
        for wrapped in wrapped_lines:
            draw_line_with_highlights(wrapped, char_index + offset_in_line)
            offset_in_line += len(wrapped)
        char_index += len(line)
        y -= 4  # spacing after each logical line

    c.save()
    buffer.seek(0)

    return dcc.send_bytes(buffer.read(), filename="annotated_text.pdf")

@app.server.route("/download-pdf")
def download_pdf():
    """
    Flask route handler that generates and returns a downloadable annotated PDF.

    Inputs are passed as query parameters:
        - 'text' (str): The text content to render in the PDF.
        - 'spans' (str): A JSON-encoded list of span dictionaries with 'start' and 'end' indices.

    Returns:
        flask.Response: A file response that triggers the download of the annotated PDF.
    """

    # Inputs
    text = request.args.get("text", "")
    spans = json.loads(request.args.get("spans", "[]"))

    # Set up PDF
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_file.name, pagesize=A4)
    width, height = A4
    x_margin = inch
    y = height - inch
    font_size = 11
    line_height = 16
    max_line_width = width - 2 * x_margin
    font_name = "Helvetica"
    c.setFont(font_name, font_size)

    # Mark spans by index
    span_lookup = {i: span for span in spans for i in range(span["start"], span["end"])}

    def draw_line_with_highlights(line_text, global_char_offset):
        """
        Draws a line of text to the PDF canvas, highlighting segments that fall within annotated spans.

        Args:
            line_text (str): The line of text to render.
            global_char_offset (int): The character offset of the line within the entire document.

        Returns:
            None
        """

        # Set initial x position and prepare to iterate through the line
        nonlocal y
        x = x_margin
        i = 0

        # Iterate through the line text, checking each character against the span lookup
        while i < len(line_text):
            # Determine if the current character is highlighted and find the extent of the highlight
            current_is_highlight = (global_char_offset + i) in span_lookup
            j = i
            while j < len(line_text) and ((global_char_offset + j) in span_lookup) == current_is_highlight:
                j += 1
            substring = line_text[i:j]
            text_width = stringWidth(substring, font_name, font_size)

            if current_is_highlight:
                c.setFillColor(HexColor("#127bf3"))
                c.rect(x - 1, y - 3, text_width + 2, line_height + 2, fill=True, stroke=False)
                c.setFillColor(white)
            else:
                c.setFillColor(black)

            c.drawString(x, y, substring)
            x += text_width
            i = j

        y -= line_height
        if y < inch:
            c.showPage()
            c.setFont(font_name, font_size)
            y = height - inch

    # For each paragraph in the text, wrap lines and draw them with highlights
    char_index = 0
    lines = text.splitlines(keepends=True)  # Keeps newlines

    for line in lines:
        line_content = line.rstrip('\r\n')
        wrapped_lines = textwrap.wrap(line_content, width=100) or ['']
        offset_in_line = 0
        for wrapped in wrapped_lines:
            draw_line_with_highlights(wrapped, char_index + offset_in_line)
            offset_in_line += len(wrapped)
        char_index += len(line)
        y -= 4  # spacing after each logical line

    c.save()
    return send_file(tmp_file.name, as_attachment=True, download_name="annotated_text.pdf")

if __name__ == '__main__':
    Timer(1, lambda: webbrowser.open("http://127.0.0.1:8050")).start()
    serve(app.server, host='127.0.0.1', port=8050)
