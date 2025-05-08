import base64
import io
import fitz  # PyMuPDF
import dash
from dash import html, dcc, Input, Output, State, Dash, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import time #can be removed after finishing

external_stylesheets = [dbc.themes.CYBORG]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MannaertsAppels Dashboard"

# Import the train_classifier function from the train_model.py file
from train_models import train_classifier

# PDF text extraction function
def extract_text_from_pdf(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

# App layout
app.layout = dbc.Container([
    # Dashboard Title
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

    # Button row
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=dbc.Button("Select PDF File", color="primary", className="w-100"),
                multiple=False
            )
        ], width=3),
        dbc.Col([
            dbc.Button("Load Classifier Model", color="info", className="w-100")
        ], width=3),
        dbc.Col([
            dbc.Button("Load SpanCat Model", color="info", className="w-100")
        ], width=3),
        dbc.Col([
            dbc.Button("Re-train Models", id="retrain-button", color="danger", className="w-100")
        ], width=3),
    ], className="mb-4"),

    # Modal for confirmation
    dbc.Modal([
        dbc.ModalHeader("Confirm Retraining"),
        dbc.ModalBody("This may take a while. Are you sure you want to proceed?"),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="cancel-retrain", color="secondary", className="me-2"),
            dbc.Button("Confirm", id="confirm-retrain", color="danger")
        ])
    ], id="retrain-modal", is_open=False),

    # Output: Extracted text
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Extracted Text"),
                dbc.CardBody([
                    dcc.Textarea(
                        id='pdf-text-output',
                        style={'width': '100%', 'height': '400px'},
                        readOnly=True
                    )
                ])
            ])
        ], width=12)
    ]),

    # Output: Retrain status with loading spinner
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

# Callback: Open/Close modal
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

# Callback: Run training after confirmation
@app.callback(
    Output("retrain-status", "children"),
    Input("confirm-retrain", "n_clicks"),
    prevent_initial_call=True
)
def retrain_model(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    try:
        status = train_classifier()
        return html.Span(status, style={"color": "limegreen", "fontWeight": "bold"})
    except Exception as e:
        return html.Span(f"Training failed: {str(e)}", style={"color": "red", "fontWeight": "bold"})

if __name__ == '__main__':
    app.run(debug=True)