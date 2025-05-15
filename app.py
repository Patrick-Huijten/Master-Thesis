import base64
import io
import fitz  # PyMuPDF
import dash
import joblib
import pandas as pd
import plotly.graph_objects as go
import re
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
                    dbc.Button("Load SpanCat Model", color="info", className="w-100", size="sm") 
                ], width=3),
                dbc.Col([ 
                    dbc.Button("Re-train Models", id="retrain-button", color="danger", className="w-100", size="sm") 
                ], width=3),
            ], className="mb-3"), 

            # Future visualizations placeholder
            # html.Div(id="future-content-placeholder", style={"height": "400px"}),

            html.Div(id="sector_pred", style={"marginTop": "50px"}),
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

#     # Modal for selecting Classifier
#     dbc.Modal([
#     dbc.ModalHeader("Select Classifier Model"),
#     dbc.ModalBody([
#         dcc.Upload(
#             id='upload-model',
#             children=dbc.Button("Select Model File", color="primary", className="w-100", size="sm"),
#             multiple=False,
#             accept=".joblib"
#         ),
#         html.Div(id="model-file-name", style={"marginTop": "10px", "color": "#00ff00"}),
#     ]),
#     dbc.ModalFooter([
#         dbc.Button("Cancel", id="cancel-model-select", color="secondary", className="me-2"),
#         dbc.Button("Confirm", id="confirm-model-select", color="primary"),
#     ]),
# ], id="model-select-modal", is_open=False),

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

# Callback: PDF extraction
@app.callback(
    Output('pdf-text-output', 'value'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_output(contents):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        text = extract_text_from_pdf(decoded)
        return text
    return ""

# Callback: Toggle modal open/close
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
        df_train = preprocess_training_data()
        status = train_classifier(df_train)
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
#________________________________________________________________________________

#Load the classification model, perform classification and plot the results
@app.callback(
    Output("sector_pred", "children"),
    Output("classification_plot", "figure"),
    Input("upload-model", "contents"),
    State("upload-data", "contents"),
    prevent_initial_call=True
)
def perform_classification(model_contents, pdf_contents):
    if model_contents is None or pdf_contents is None:
        raise PreventUpdate

    # # Decode and save the uploaded model file temporarily
    # model_content_type, model_content_string = model_contents.split(',')
    # model_bytes = base64.b64decode(model_content_string)
    # temp_model_path = "temp_uploaded_model.joblib"
    # with open(temp_model_path, "wb") as f:
    #     f.write(model_bytes)

    # Helper function to decode base64 contents
    def decode_contents(contents):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return io.BytesIO(decoded)

    # Decode and load the model
    model_file = decode_contents(model_contents)

    try:
        classifier = joblib.load(model_file)
    except Exception as e:
        print("Something went wrong while loading the model")
        return f"Error loading model: {e}", {}

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
    fig = go.Figure()
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

    return prediction_display, fig

if __name__ == '__main__':
    app.run(debug=True)