from dash import Dash, html
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.SLATE]

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Mental Health Dataset Analysis."

app.layout = html.Div([
    html.H1("Hello World")
])

if __name__ == '__main__':
    app.run(debug=True)