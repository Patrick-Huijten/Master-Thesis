import base64
import io
import fitz  # PyMuPDF
import dash
from datetime import datetime
import joblib
import pandas as pd
import plotly.graph_objects as go
import re
import os
import matplotlib.pyplot as plt
from dash import html, dcc, Input, Output, State, Dash, ctx, clientside_callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# External stylesheet
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MannaertsAppels Dashboard"
app.server.max_content_length = 1024 * 1024 * 1000  # 1GB limit

# Import training logic
from Sector_Classification import train_classifier, preprocess_training_data, classify
from SpanCat_code import SpanCat_data_prep, train_SpanCat, predict_spans, highlight_spans, editable_highlight_spans

# PDF text extraction
def extract_text_from_pdf(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

# App layout
app.layout = dbc.Container([

    # Save the spans found by SpanCat for reference
    dcc.Store(id="current-spans", data=[]),
    dcc.Store(id="current-text", data=""),
    dcc.Store(id="selected-text-store", data={"start": None, "end": None}),

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
                    # dcc.Textarea( 
                    #     id='pdf-text-output',
                    #     style={'width': '100%', 'height': 550, 'fontSize': '12px'}, 
                    #     readOnly=True
                    # ),
                    html.Div(
                        "Upload a PDF file using the button on the right. The contents will appear here.",
                        id='highlighted-text-display',
                        style={'height': '550px', 'overflowY': 'scroll', 'whiteSpace': 'pre-wrap', 'fontSize': '20px'}
                    ),
                    html.Div([ 
                        "Font Size", 
                        dcc.Slider( 
                            id="font-size-slider",
                            min=8, 
                            max=36, 
                            step=1, 
                            value=20,  # Default value
                            marks={i: f"{i}" for i in range(8, 37, 4)},  # Mark every 4 units
                        ),
                        dbc.Button("Add Selected Span", id="add-span-btn", color="success", className="mt-2")
                    ], style={"marginTop": "20px"})
                ]) 
            ]) 
        ], width=5), 

        # Right column: Buttons + visualization space
        dbc.Col([ 
            dbc.Row([ 
                dbc.Col([ 
                    dcc.Upload( 
                        id='upload-data', 
                        children=dbc.Button("Select PDF File", color="primary", className="w-100", size="sm"), 
                        multiple=False
                    )
                ], width=3),
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

                # Right: dropdown + button vertically stacked
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
                        "Add to training dataset",
                        id="save-pdf-sector",
                        color="success",
                        className="w-100",
                        size="sm"
                    )
                ], width=4)
            ], className="mb-4", align="start"),

            dcc.Graph(figure={}, id="classification_plot", style={"height": "400px"}),
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

    # Modal for selecting SpanCat model
    dbc.Modal([
    # dbc.ModalHeader("Select Classifier Model"),
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

    # Loading spinner and retrain status
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
    if not n_clicks:
        raise PreventUpdate
    try:
        # Pre-process training data and train the classifier
        df_train = preprocess_training_data()
        status = train_classifier(df_train)

        # Prepare data for SpanCat and train the model
        SpanCat_data_prep()
        status += train_SpanCat()

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
    Input("upload-model", "contents"),
    Input("upload-data", "contents"),
    Input("confirm-model-select", "n_clicks"),
    Input({'type': 'highlighted-span', 'index': dash.ALL}, 'n_clicks'),
    Input("selected-text-store", "data"),  # TRIGGER for adding spans
    State("SpanCat-specialization-radio", "value"),
    State("current-spans", "data"),
    State("current-text", "data"),
    prevent_initial_call=True
)
def unified_handler(
    classifier_contents,
    pdf_contents,
    confirm_spancat_clicks,
    span_clicks,
    selection,
    specialization,
    spans,
    text
):
    fig = go.Figure()
    prediction_display = None
    triggered_id = ctx.triggered_id

    # Case 1: Remove a span
    if isinstance(triggered_id, dict) and triggered_id.get("type") == "highlighted-span":
        index_to_remove = triggered_id.get("index")
        new_spans = spans[:index_to_remove] + spans[index_to_remove + 1:]
        new_components = editable_highlight_spans(text, new_spans)
        return new_components, new_spans, text, dash.no_update, dash.no_update

    # Case 2: Add a new span (triggered by selected-text-store update)
    if triggered_id == "selected-text-store":
        if not selection or selection.get("text") is None:
            raise PreventUpdate

        selected_text = selection["text"]
        start = text.find(selected_text)
        end = start + len(selected_text)

        if start == -1 or start == end or end > len(text):
            raise PreventUpdate

        new_span = {
            "text": selected_text,
            "start": start,
            "end": end,
            "label": specialization,
            "score": 1.0
        }

        updated_spans = spans + [new_span]
        new_components = editable_highlight_spans(text, updated_spans)
        return new_components, updated_spans, text, dash.no_update, dash.no_update

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
            return text, [], text, None, fig

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

    # Case 5: Predict spans via SpanCat model
    spans = []
    highlighted = None
    if pdf_contents and confirm_spancat_clicks:
        spans = predict_spans(specialization, text)
        highlighted = editable_highlight_spans(text, spans)

    if text and not highlighted:
        highlighted = text

    return highlighted, spans, text, prediction_display, fig

@app.callback(
    Output("retrain-status", "children", allow_duplicate=True),
    Input("save-pdf-sector", "n_clicks"),
    State("true-class-dropdown", "value"),
    State("upload-data", "contents"),
    prevent_initial_call=True
)
def save_pdf_to_class_folder(n_clicks, selected_class, pdf_contents):
    if not pdf_contents or not selected_class:
        raise PreventUpdate

    try:
        base_dir = "Classifier data"
        class_dir = os.path.join(base_dir, selected_class)
        os.makedirs(class_dir, exist_ok=True)

        # Find highest existing ID (X)
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
    
# @app.callback(
#     Output("current-spans", "data"),
#     Output("highlighted-text-display", "children"),
#     Input({'type': 'highlighted-span', 'index': dash.ALL}, 'n_clicks'),
#     State("current-spans", "data"),
#     State("current-text", "data"),
#     prevent_initial_call=True
# )
# def remove_span(n_clicks_list, spans, text):
#     triggered_index = next((i for i, n in enumerate(n_clicks_list) if n), None)
#     if triggered_index is None:
#         raise PreventUpdate

#     updated_spans = spans[:triggered_index] + spans[triggered_index+1:]
#     components = editable_highlight_spans(text, updated_spans)
#     return updated_spans, components

if __name__ == '__main__':
    app.run(debug=True)