import base64
import io
import fitz  # PyMuPDF
import dash
from dash import html, dcc, Input, Output, State, Dash, ctx
import dash_bootstrap_components as dbc

#Initialize the Dash app
external_stylesheets = [dbc.themes.CYBORG]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MannaertsAppels Dashboard"

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
                    html.Div([  # **Using 'html.Div' for custom styling**
                        html.Span("Mannaerts", style={'color': '#72787c', 'fontWeight': 'bold', 'fontSize': '36px'}),  # **Custom styling for "Mannaerts"**
                        html.Span("Appels", style={'color': '#ea1a8d', 'fontWeight': 'bold', 'fontSize': '36px'})  # **Custom styling for "Appels"**
                    ], className="text-center")  # **Centering the title**
                )
            ], className="mb-4"),
            width=12
        )
    ]),

    # File Upload Button + Additional Buttons in the same row
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=dbc.Button("Select PDF File", color="primary", className="w-100"),
                multiple=False
            )
        ], width=3),  # Making this button take up 3 columns

        dbc.Col([
            dbc.Button("Load Classifier Model", color="info", className="w-100")
        ], width=3),  # Making this button take up 3 columns

        dbc.Col([
            dbc.Button("Load SpanCat Model", color="info", className="w-100")
        ], width=3),  # Making this button take up 3 columns

        dbc.Col([
            dbc.Button("Re-train Models", color="danger", className="w-100")
        ], width=3),  # Making this button take up 3 columns
    ], className="mb-4"),

    # Text Output Area
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
    ])
], fluid=True)

# Callback for processing uploaded PDF
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

if __name__ == '__main__':
    app.run(debug=True)