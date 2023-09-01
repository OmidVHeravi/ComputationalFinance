import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import base64
import io
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Reinforcement Learning for Trading"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div([
                'Unsupported file type uploaded'
            ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # Check if 'BuySellRate' column exists in the DataFrame
    if 'BuySellRate' not in df.columns:
        return html.Div([
            'The column "BuySellRate" does not exist in the uploaded file.'
        ])

    # Filter rows with NaN, 0, or 1 in the BuySellRate column
    df = df[~df['BuySellRate'].isin([np.nan, 0, 1])]

    return html.Div([
        html.H5(filename),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),
        html.Hr(),  # horizontal line
    ])

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              Input('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)

