import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from itertools import combinations
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load data
df = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = df.apply(lambda row: set(
    filter(pd.notna, row)), axis=1).tolist()

DEFAULT_SUPPORT_LEVELS = [0.1, 0.05, 0.01, 0.005]
DEFAULT_CONFIDENCE_LEVELS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
DEFAULT_TOP_ITEMS = 10


def supp(X, Y, transactions):
    total = len(transactions)
    count = sum(1 for t in transactions if (X | Y).issubset(t))
    return count / total if total > 0 else 0.0


def conf(X, Y, transactions):
    supp_X = supp(X, set(), transactions)
    supp_XY = supp(X, Y, transactions)
    return supp_XY / supp_X if supp_X > 0 else 0.0


def lift(X, Y, transactions):
    supp_X = supp(X, set(), transactions)
    supp_Y = supp(Y, set(), transactions)
    supp_XY = supp(X | Y, set(), transactions)
    return supp_XY / (supp_X * supp_Y)


def conv(X, Y, transactions):
    supp_Y = supp(Y, set(), transactions)
    conf_XY = conf(X, Y, transactions)
    return (1-supp_Y)/(1-conf_XY)


def generate_simple_rules(top_items):
    rules = []
    for X in top_items:
        for Y in top_items:
            if X != Y:
                X_set, Y_set = {X}, {Y}
                s = supp(X_set, Y_set, transactions)
                c = conf(X_set, Y_set, transactions)
                rules.append((X, Y, s, c))
    return rules


def apriori_gen(L_prev, k):
    Ck = set()
    L_prev = list(L_prev)
    for i in range(len(L_prev)):
        for j in range(i + 1, len(L_prev)):
            l1, l2 = sorted(L_prev[i]), sorted(L_prev[j])
            if l1[:k-2] == l2[:k-2]:
                candidate = frozenset(set(l1) | set(l2))
                subsets = combinations(candidate, k - 1)
                if all(frozenset(s) in L_prev for s in subsets):
                    Ck.add(candidate)
    return Ck


def apriori(transactions, min_support, min_confidence, top_n):
    item_counts = Counter(item for t in transactions for item in t)
    top_items = [item for item, _ in item_counts.most_common(top_n)]
    filtered_transactions = [t & set(top_items)
                             for t in transactions if t & set(top_items)]

    num_transactions = len(filtered_transactions)
    item_counts = Counter(item for t in filtered_transactions for item in t)
    L1 = set(frozenset([item]) for item, count in item_counts.items()
             if count / num_transactions >= min_support)

    L = []
    Lk = L1
    k = 2
    while Lk:
        L.append(Lk)
        Ck = apriori_gen(Lk, k)
        candidate_counts = defaultdict(int)

        for t in filtered_transactions:
            for candidate in Ck:
                if candidate.issubset(t):
                    candidate_counts[candidate] += 1

        Lk = set()
        for candidate, count in candidate_counts.items():
            if count / num_transactions >= min_support:
                Lk.add(candidate)

        k += 1

    rules = []
    for level in L:
        for itemset in level:
            if len(itemset) < 2:
                continue
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    if not consequent:
                        continue
                    s = supp(set(antecedent), set(
                        consequent), filtered_transactions)
                    c = conf(set(antecedent), set(
                        consequent), filtered_transactions)
                    l = lift(set(antecedent), set(
                        consequent), filtered_transactions)
                    v = conv(set(antecedent), set(
                        consequent), filtered_transactions)
                    if s >= min_support and c >= min_confidence:
                        rules.append({
                            "antecedent": ", ".join(sorted(antecedent)),
                            "consequent": ", ".join(sorted(consequent)),
                            "support": round(s, 4),
                            "confidence": round(c, 4),
                            "lift": round(l, 4),
                            "conviction": round(v, 4)
                        })
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
    html.H2("Association Rule Explorer"),

    html.H5("Support Levels:"),
    dbc.InputGroup([
        dbc.Input(id="support-input", type="number", min=0, max=1,
                  step="any", placeholder="Enter support level"),
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

    html.H5("Number of Top Items to Use:"),
    dbc.Input(id="top-n-items", type="number", min=1,
              max=100, step=1, value=DEFAULT_TOP_ITEMS),

    dcc.Graph(id="simple-rules-graph"),
    html.H5("1-1 Rules Table:"),
    dash_table.DataTable(
        id="simple-rules-table",
        columns=[
            {"name": "Antecedent", "id": "antecedent"},
            {"name": "Consequent", "id": "consequent"},
            {"name": "Support", "id": "support"},
            {"name": "Confidence", "id": "confidence"},
        ],
        sort_action="native",
        page_size=10
    ),

    html.Hr(),
    html.H5("Apriori Thresholds"),
    dbc.Input(id="apriori-support", type="number", min=0,
              max=1, step=0.001, placeholder="Support level"),
    dbc.Input(id="apriori-confidence", type="number", min=0,
              max=1, step=0.001, placeholder="Confidence level"),
    html.Br(),
    dbc.Button("Run Apriori", id="run-apriori", color="success"),
    html.Br(),
    dcc.Graph(id="apriori-graph"),
    html.H5("Apriori Rules Table:"),
    dash_table.DataTable(
        id="apriori-table",
        columns=[
            {"name": "Antecedent", "id": "antecedent"},
            {"name": "Consequent", "id": "consequent"},
            {"name": "Support", "id": "support"},
            {"name": "Confidence", "id": "confidence"},
            {"name": "Lift", "id": "lift"},
            {"name": "Conviction", "id": "conviction"}
        ],
        sort_action="native",
        page_size=10
    ),

    dcc.Store(id="stored-support", data=DEFAULT_SUPPORT_LEVELS),
    dcc.Store(id="stored-confidence", data=DEFAULT_CONFIDENCE_LEVELS),
])


@app.callback(
    Output("simple-rules-graph", "figure"),
    Output("simple-rules-table", "data"),
    Input("stored-support", "data"),
    Input("stored-confidence", "data"),
    Input("top-n-items", "value")
)
def update_simple_rules(support_levels, confidence_levels, top_n):
    item_counts = Counter(item for t in transactions for item in t)
    top_items = [item for item, _ in item_counts.most_common(top_n)]
    simple_rules = generate_simple_rules(top_items)

    traces = []
    table_data = []
    for s_level in support_levels:
        x_vals = []
        y_vals = []
        for c_level in confidence_levels:
            count = sum(1 for _, _, s, c in simple_rules if s >=
                        s_level and c >= c_level)
            x_vals.append(c_level)
            y_vals.append(count)
            for X, Y, s, c in simple_rules:
                if s >= s_level and c >= c_level:
                    table_data.append({"antecedent": X, "consequent": Y, "support": round(
                        s, 4), "confidence": round(c, 4)})
        traces.append(go.Scatter(x=x_vals, y=y_vals,
                      mode="lines+markers", name=f"Support ≥ {s_level}"))

    figure = go.Figure(data=traces, layout=go.Layout(
        title="1-1 Rules Count vs Confidence", xaxis={"title": "Confidence"}, yaxis={"title": "Rule Count"}))
    return figure, table_data


@app.callback(
    Output("apriori-graph", "figure"),
    Output("apriori-table", "data"),
    Input("run-apriori", "n_clicks"),
    State("apriori-support", "value"),
    State("apriori-confidence", "value"),
    State("top-n-items", "value"),
    prevent_initial_call=True
)
def run_apriori(n_clicks, supp_lvl, conf_lvl, top_n):
    rules = apriori(transactions, float(supp_lvl),
                    float(conf_lvl), int(top_n))
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[r["support"] for r in rules],
                y=[r["confidence"] for r in rules],
                mode="markers",
                text=[f"{r['antecedent']} ⇒ {r['consequent']}" for r in rules],
                hoverinfo="text+x+y"
            )
        ],
        layout=go.Layout(title="Apriori Rules", xaxis={
            "title": "Support"}, yaxis={"title": "Confidence"})
    )
    return fig, rules


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
