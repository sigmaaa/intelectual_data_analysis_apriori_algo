import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from itertools import combinations
import pandas as pd
from collections import Counter

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

df = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = df.apply(lambda row: set(
    filter(pd.notna, row)), axis=1).tolist()

DEFAULT_SUPPORT_LEVELS = [0.1, 0.05, 0.01, 0.005]
DEFAULT_CONFIDENCE_LEVELS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def supp(X, Y, transactions):
    total = len(transactions)
    count = sum(1 for t in transactions if (X | Y).issubset(t))
    print(count)
    print(total)
    return count / total if total > 0 else 0.0


def conf(X, Y, transactions):
    supp_X = sum(1 for t in transactions if X.issubset(t)) / len(transactions)
    supp_XY = supp(X, Y, transactions)
    return supp_XY / supp_X if supp_X > 0 else 0.0


def generate_rules(items):
    rules = []
    for X in combinations(items, 1):
        X = set(X)
        for y in set(items) - X:
            Y = {y}
            rules.append((X, Y))
    return rules


def create_list_items(values, id_prefix):
    return [
        html.Li([
            f"{v:.3f} ",
            dbc.Button("x", id={"type": id_prefix, "index": i},
                       size="sm", color="danger", className="ms-2")
        ]) for i, v in enumerate(values)
    ]


app.layout = dbc.Container([
    html.H2("Association Rule Support/Confidence Visualizer"),

    html.H5("Support Levels:"),
    dbc.InputGroup([
        dbc.Input(id="support-input", type="number", min=0, max=1,
                  step=0.01, placeholder="Enter support level"),
        dbc.Button("+", id="add-support", color="primary")
    ], className="mb-2"),
    html.Ul(id="support-list",
            children=create_list_items(DEFAULT_SUPPORT_LEVELS, "remove-support")),

    html.H5("Confidence Levels:"),
    dbc.InputGroup([
        dbc.Input(id="confidence-input", type="number", min=0, max=1,
                  step=0.01, placeholder="Enter confidence level"),
        dbc.Button("+", id="add-confidence", color="primary")
    ], className="mb-2"),
    html.Ul(id="confidence-list",
            children=create_list_items(DEFAULT_CONFIDENCE_LEVELS, "remove-confidence")),

    dcc.Graph(id="rules-graph"),

    html.H5("Generated Rules Table:"),
    dash_table.DataTable(
        id="rules-table",
        columns=[
            {"name": "Antecedent", "id": "antecedent"},
            {"name": "Consequent", "id": "consequent"},
            {"name": "Support", "id": "support", "type": "numeric"},
            {"name": "Confidence", "id": "confidence", "type": "numeric"}
        ],
        sort_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left"}
    ),

    dcc.Store(id="stored-support", data=DEFAULT_SUPPORT_LEVELS),
    dcc.Store(id="stored-confidence", data=DEFAULT_CONFIDENCE_LEVELS),

])


@app.callback(
    Output("rules-graph", "figure"),
    Output("rules-table", "data"),
    Input("stored-support", "data"),
    Input("stored-confidence", "data"),

)
def update_graph_and_table(support_levels, confidence_levels):
    used_transactions = transactions

    item_counts = Counter(i for t in transactions for i in t)
    top_items = [item for item, _ in item_counts.most_common(10)]
    rules = generate_rules(top_items)

    traces = []
    table_data = []

    for s_level in support_levels:
        x_vals = []
        y_vals = []
        for c_level in confidence_levels:
            count = 0
            for X, Y in rules:
                s = supp(X, Y, used_transactions)
                c = conf(X, Y, used_transactions)
                if s >= s_level and c >= c_level:
                    count += 1
                    table_data.append({
                        "antecedent": ", ".join(sorted(X)),
                        "consequent": ", ".join(sorted(Y)),
                        "support": round(s, 4),
                        "confidence": round(c, 4)
                    })
            x_vals.append(c_level)
            y_vals.append(count)
        traces.append(go.Scatter(x=x_vals, y=y_vals,
                      mode="lines+markers", name=f"Support ≥ {s_level:.3f}"))

    figure = go.Figure(
        data=traces,
        layout=go.Layout(
            title="Number of Rules vs Confidence Level",
            xaxis={"title": "Confidence Level"},
            yaxis={"title": "Number of Rules"},
            hovermode="closest"
        )
    )

    return figure, table_data


@app.callback(
    Output("stored-support", "data"),
    Output("support-list", "children"),
    Input("add-support", "n_clicks"),
    Input({"type": "remove-support", "index": dash.ALL}, "n_clicks"),
    State("support-input", "value"),
    State("stored-support", "data"),
    prevent_initial_call=True
)
def update_support(add_click, remove_clicks, value, stored):
    triggered_id = ctx.triggered_id
    if isinstance(triggered_id, dict):
        idx = triggered_id["index"]
        stored.pop(idx)
    elif value is not None and value not in stored:
        stored.append(value)
    items = create_list_items(stored, "remove-support")
    return stored, items


@app.callback(
    Output("stored-confidence", "data"),
    Output("confidence-list", "children"),
    Input("add-confidence", "n_clicks"),
    Input({"type": "remove-confidence", "index": dash.ALL}, "n_clicks"),
    State("confidence-input", "value"),
    State("stored-confidence", "data"),
    prevent_initial_call=True
)
def update_confidence(add_click, remove_clicks, value, stored):
    triggered_id = ctx.triggered_id
    if isinstance(triggered_id, dict):
        idx = triggered_id["index"]
        stored.pop(idx)
    elif value is not None and value not in stored:
        stored.append(value)
    items = create_list_items(stored, "remove-confidence")
    return stored, items


if __name__ == "__main__":
    app.run(debug=True)
