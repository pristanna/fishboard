import functools
import logging
import pathlib

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import plotly.graph_objects as go

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


METRICS = {"MAE": mean_absolute_error, "MSE": mean_squared_error, "R2": r2_score}


def main():
    st.title("Fishboard")

    st.header("Načtení dat")
    fish_data = pd.read_csv(DATA_DIR.joinpath("fish_data.csv"), index_col=0)
    learning_data = fish_data.drop(columns=["ID"])  # zahodim sloupec ID
    learning_data = pd.get_dummies(learning_data)  # převedu kategorické proměnné
    st.write(learning_data)

    st.header("Výběr odezvy")
    target = st.selectbox("Sloupec s odezvou", learning_data.columns)
    y = learning_data[target]
    X = learning_data.drop(columns=[target])

    st.header("Rozdělení na testovací a trénovací data")
    test_size = st.slider("Poměr testovací sady", 0.0, 1.0, 0.25, 0.05)
    stratify_column = st.selectbox("Stratify", [None] + list(fish_data.columns))
    if stratify_column is not None:
        stratify = fish_data[stratify_column]
    else:
        stratify = None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify)

    st.header("Výběr modelu")
    model = st.selectbox("Regresní model", list(REGRESSION_MODELS))
    hyperparams = {
        hyperparam: widget() for hyperparam, widget in REGRESSION_MODELS[model]["hyperparams"].items()
    }

    regressor = REGRESSION_MODELS[model]["class"](**hyperparams)
    regressor.fit(X_train, y_train)

    metric = st.selectbox("Metrika", list(METRICS))
    y_predicted = regressor.predict(X_test)
    error = METRICS[metric](y_predicted, y_test)
    st.write(f"{metric}: {error:.3g}")

    st.header("Vizualizace")

    predicted_target = f"{target} - predicted"
    complete_data = fish_data.assign(**{predicted_target: regressor.predict(X)})
    fig = px.scatter(complete_data, x=target, y=predicted_target, color="Species")
    fig.add_trace(
        go.Scatter(
            x=[complete_data[target].min(), complete_data[target].max()],
            y=[complete_data[target].min(), complete_data[target].max()],
            mode="lines",
            line=dict(width=2, color="DarkSlateGrey"),
            name="ideal prediction"
        )
    )
    st.write(fig)


if __name__ == "__main__":
    logging.basicConfig()
    main()
