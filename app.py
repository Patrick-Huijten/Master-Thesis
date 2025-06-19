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
from dash import html, dcc, Input, Output, State, Dash, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# External stylesheet
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MannaertsAppels Dashboard"
app.server.max_content_length = 1024 * 1024 * 1000  # 1GB limit

# Import training logic
from Sector_Classification import train_classifier, preprocess_training_data, classify
from SpanCat_code import SpanCat_data_prep, train_SpanCat, predict_spans

# PDF text extraction
def extract_text_from_pdf(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

# App layout
app.layout = dbc.Container([

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
                    dcc.Textarea( 
                        id='pdf-text-output',
                        style={'width': '100%', 'height': 550, 'fontSize': '12px'}, 
                        readOnly=True
                    ),
                    html.Div([ 
                        "Font Size", 
                        dcc.Slider( 
                            id="font-size-slider",
                            min=8, 
                            max=36, 
                            step=1, 
                            value=12,  # Default value
                            marks={i: f"{i}" for i in range(8, 37, 4)},  # Mark every 4 units
                        ) 
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
    Output("pdf-text-output", "style"),
    Input("font-size-slider", "value")
)
def update_font_size(font_size):
    return {'width': '100%', 'height': 550, 'fontSize': f'{font_size}px'}

#Load the classification model, perform classification and plot the results
@app.callback(
    Output('pdf-text-output', 'value'),
    Output("sector_pred", "children"),
    Output("classification_plot", "figure"),
    Input("upload-model", "contents"),
    Input("upload-data", "contents"),
    Input("confirm-model-select", "n_clicks"),
    State("SpanCat-specialization-radio", "value"),
    prevent_initial_call=True
)
def handle_uploads(classifier_contents, pdf_contents, confirm_SpanCat, SpanCat_specialization): # add SpanCat_contents
    if classifier_contents is None and pdf_contents is None:
        raise PreventUpdate
    
    fig = go.Figure()
    text = ""
    prediction_display = None

    if pdf_contents:
        content_type, content_string = pdf_contents.split(',')
        decoded = base64.b64decode(content_string)
        text = extract_text_from_pdf(decoded)

    # If a pdf has been uploaded and a classifier has been selected, predict the class of the pdf and visualize the results
    if classifier_contents and pdf_contents:

        # Helper function to decode base64 contents
        def decode_contents(contents):
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            return io.BytesIO(decoded)

        # Decode and load the model
        model_file = decode_contents(classifier_contents)

        try:
            classifier = joblib.load(model_file)
        except Exception as e:
            print("Something went wrong while loading the model")
            return text, None, fig

        # Extract text from the uploaded PDF
        content_type, content_string = pdf_contents.split(',')
        decoded = base64.b64decode(content_string)
        text = extract_text_from_pdf(decoded)

        # Split text into paragraphs
        paragraphs = [para.strip() for para in re.split(r'\n\s*\n', text) if para.strip()]
        df_new = pd.DataFrame({"text": paragraphs})

        # Pass the temp model path to your classify function
        result_string, df_class_probas = classify(df_new, classifier)

        #Generate a stacked bar chart of the classification probabilities
        class_mapping = {
            0: 'Bouw & Vastgoed',
            1: 'Handel & Industrie',
            2: 'Zakelijke Dienstverlening',
            3: 'Zorg'}
        
        colors = {
            'UNKNOWN — unable to determine a reliable class': '#888888',  # Grey
            'Bouw & Vastgoed': '#009E73',                                 # Green
            'Handel & Industrie': '#0072B2',                              # Blue
            'Zakelijke Dienstverlening': '#CC79A7',                       # Pink
            'Zorg': '#D55E00'                                             # Orange
    }
        
        # Convert list of probabilities into separate columns
        proba_values = df_class_probas['pred_probabilities'].tolist()

        # Ensure all probability vectors are of length 4
        assert all(len(p) == 4 for p in proba_values), "Each probability list must have 4 elements"

        # Create a DataFrame from the probability lists
        df = pd.DataFrame(proba_values).rename(columns=class_mapping)

        # Create paragraph labels for the x-axis
        paragraph_labels = [f"P{i+1}" for i in range(len(df))]

        # Build the stacked bar chart
        for class_name in df.columns:
            fig.add_trace(go.Bar(
                x=paragraph_labels,
                y=df[class_name],
                name=class_name,
                marker_color=colors.get(class_name, '#888'),  # fallback color
                hovertemplate=f"%{{x}}<br>{class_name}: %{{y:.2%}}<extra></extra>"
            ))

        fig.update_layout(
            title="Sector Probabilities Per Paragraph",
            barmode='stack',
            xaxis_title='Paragraph',
            yaxis_title='Probability',
            yaxis=dict(
                range=[0, 1],
                tickformat=".0%"),  # This formats 0.2 as 20%
            legend_title='Class',
            template='plotly_dark'  # matches CYBORG theme
        )

        color = colors.get(result_string, '#ffffff')
    #     prediction_display = dcc.Markdown(
    #         f"**Prediction:**<br><span style='color:{color}; font-weight:bold; font-size:24px'>{result_string}</span>",
    #         dangerously_allow_html=True
    # )
        prediction_display = html.Div([
            html.Span("Prediction: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.Span(result_string, style={"color": color, "fontWeight": "bold", "fontSize": "24px"})])
    
    # If a pdf has been uploaded and a SpanCat model has been selected, perform span detection and show the results
    if pdf_contents and confirm_SpanCat:
        spans = predict_spans(SpanCat_specialization, text)

    return text, prediction_display, fig

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

if __name__ == '__main__':
    app.run(debug=True)