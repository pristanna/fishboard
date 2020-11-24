import functools
import logging
import pathlib
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = pathlib.Path("data")


REGRESSION_MODELS = {
    "LinearRegression": {
        "class": LinearRegression,
        "hyperparams": {},
    },
    "Lasso": {
        "class": Lasso,
        "hyperparams": {"alpha": functools.partial(st.slider, "alpha", 0.0, 1.0, 0.0)},
    },
    "SVR": {
        "class": SVR,
        "hyperparams": {
            "kernel": functools.partial(
                st.selectbox, "kernel", ["linear", "poly", "rbf", "sigmoid"], index=2
            ),
            "C": functools.partial(st.number_input, "C", 0.0, None, 1.0),
        },
    },
}


CLASSIFIERS = {
    "DecisionTreeClassifier": {
        "class": DecisionTreeClassifier,
        "hyperparams": {},
    },
}


METRICS = {"MAE": mean_absolute_error, "MSE": mean_squared_error, "R2": r2_score}


@st.cache
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR.joinpath("fish_data.csv"), index_col=0)


@st.cache
def preprocess(
    data: pd.DataFrame, drop_columns: Optional[List] = None, get_dummies: bool = False
) -> pd.DataFrame:
    if drop_columns:
        data = data.drop(columns=drop_columns)
    if get_dummies:
        data = pd.get_dummies(data)
    return data


def regression(col1, col2, fish_data, target, X, X_train, X_test, y_train, y_test):
    expander = col1.beta_expander("Výběr modelu")
    with expander:
        model = st.selectbox("Regresní model", list(REGRESSION_MODELS))
        hyperparams = {
            hyperparam: widget() for hyperparam, widget in REGRESSION_MODELS[model]["hyperparams"].items()
        }
        metric = st.selectbox("Metrika", list(METRICS))

    regressor = REGRESSION_MODELS[model]["class"](**hyperparams)
    try:
        regressor.fit(X_train, y_train)
    except Exception as error:
        st.error(f"Chyba při fitování modelu: {error}")
        return
    y_predicted = regressor.predict(X_test)
    error = METRICS[metric](y_predicted, y_test)

    col2.header("Výsledky modelu")
    col2.write(f"{metric}: {error:.3g}")

    predicted_target = f"{target} - predicted"
    # X = pd.concat((X_train, X_test), axis=0)
    complete_data = fish_data.assign(**{predicted_target: regressor.predict(X)})
    fig = px.scatter(complete_data, x=target, y=predicted_target, color="Species")
    fig.add_trace(
        go.Scatter(
            x=[complete_data[target].min(), complete_data[target].max()],
            y=[complete_data[target].min(), complete_data[target].max()],
            mode="lines",
            line=dict(width=2, color="DarkSlateGrey"),
            name="ideal prediction",
        )
    )
    col2.write(fig)


def classification(col1, col2, fish_data, target, X, X_train, X_test, y_train, y_test):
    st.error("Tohle ještě chybí")


def main():
    st.set_page_config(page_title="Fishboard", layout="wide")
    st.title("Fishboard")
    col1, col2 = st.beta_columns(2)

    col1.header("Načtení dat")
    fish_data = load_data()
    with col1.beta_expander("Preprocessing"):
        drop_columns = st.multiselect("Drop columns", fish_data.columns)
        get_dummies = st.checkbox("Get dummies")
    learning_data = preprocess(fish_data, drop_columns, get_dummies)

    with col1.beta_expander("Zobrazení dat"):
        st.dataframe(learning_data, height=150)
        fig = px.scatter_matrix(learning_data)
        st.write(fig)

    target = col1.selectbox("Sloupec s odezvou", learning_data.columns)
    y = learning_data[target]
    X = learning_data.drop(columns=[target])

    # col1.header("Rozdělení na testovací a trénovací data")
    expander = col1.beta_expander("Rozdělení na testovací a trénovací data")
    with expander:
        test_size = st.slider("Poměr testovací sady", 0.0, 1.0, 0.25, 0.05)
        stratify_column = st.selectbox("Stratify", [None] + list(fish_data.columns))
    if stratify_column is not None:
        stratify = fish_data[stratify_column]
    else:
        stratify = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify)
    regression(col1, col2, fish_data, target, X, X_train, X_test, y_train, y_test)




if __name__ == "__main__":
    logging.basicConfig()
    main()
