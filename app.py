import base64
import io
import fitz  # PyMuPDF
import dash
from dash import html, dcc, Input, Output, State, Dash, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# External stylesheet
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MannaertsAppels Dashboard"

# Import training logic
from train_models import train_classifier, preprocess_training_data

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
                    dbc.Button("Load Classifier Model", color="info", className="w-100", size="sm") 
                ], width=3),
                dbc.Col([ 
                    dbc.Button("Load SpanCat Model", color="info", className="w-100", size="sm") 
                ], width=3),
                dbc.Col([ 
                    dbc.Button("Re-train Models", id="retrain-button", color="danger", className="w-100", size="sm") 
                ], width=3),
            ], className="mb-3"), 

            # Future visualizations placeholder
            html.Div(id="future-content-placeholder", style={"height": "400px"}) 
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
def toggle_modal(retrain_clicks, confirm_clicks, cancel_clicks, is_open):
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
    Output('pdf-text-output', 'style'),
    Input('font-size-slider', 'value')
)
def update_font_size(font_size):
    return {'width': '100%', 'height': 550, 'fontSize': f'{font_size}px'}

if __name__ == '__main__':
    app.run(debug=True)