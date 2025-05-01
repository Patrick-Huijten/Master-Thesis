import dash
from dash import html, dcc, Input, Output, State, Dash, ctx
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.CYBORG]

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MannaertsAppels Dashboard"

app.layout = dbc.Container([
    #Construct the title and subtitles
    dbc.Row([
        html.Div('MannaertsAppels Dashboard', style={
            'fontSize': '36px', 
            'fontWeight': 'bold', 
            'textAlign': 'center', 
            'marginBottom': '20px'
        })
    ]),

    dbc.Row([
        dcc.Upload(
            id='upload-data',
            children=dbc.Button("Select File", color="primary"),
            multiple=False,
            style={'display': 'inline-block'}
        )
    ])
])

if __name__ == '__main__':
    app.run(debug=True)