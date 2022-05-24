"""
Module that contains all of the analysis functions
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Save Model Using Pickle
from pickle import dump
from pickle import load

sns.set_theme(style="darkgrid")
sns.set_palette("colorblind")


def load_data(filename="Bike-Sharing-Dataset/hour.csv", print_diagnostics=False):
    """
    Reads the CSV data file and converts it to a pandas dataframe.
    Inputs:
        - input CSV file path
        - print_diagnostics as a boolean to show statistical and type info of the dataframe
    """
    df = pd.read_csv(filename)

    if print_diagnostics:
        print(df.head())
        print(df.describe())
        print(df.info())

    return df


def prepare_data_for_EDA(data):
    """
    Drop the unnecessary columns.
    Check for null values
    """
    df = data
    try:
        if df.isnull().values.any():
            raise ValueError("Contains NaN entries!")
    except (ValueError, IndexError):
        exit("Could not complete due to NaN entries!")

    # rename some columns
    df = df.rename(
        columns={
            "weathersit": "weather",
            "yr": "year",
            "mnth": "month",
            "hr": "hour",
            "hum": "humidity",
            "cnt": "count",
        }
    )

    df = df.drop(columns=["instant", "dteday", "year"])
    # change int columns to category
    cols = ["season", "month", "hour", "holiday", "weekday", "workingday", "weather"]
    for col in cols:
        df[col] = df[col].astype("category")

    return df


def prepare_data(data, print_diagnostics=False):
    """
    Drop the unnecessary columns.
    Perform one-hot encoding.
    """
    df = data
    df = prepare_data_for_EDA(data=df)
    df["count"] = np.log(df["count"])
    df_oh = df
    cols = ["season", "month", "hour", "holiday", "weekday", "workingday", "weather"]

    for col in cols:
        df_oh = one_hot_encoding(df_oh, col)
    if print_diagnostics:
        print(df_oh.head())

    return df_oh


def split_train_test(data, validation_fraction=0.25):
    """
    split out validation dataset for training!
    """
    df_oh = data
    X = df_oh.drop(
        columns=["atemp", "windspeed", "casual", "registered", "count"], axis=1
    )
    y = df_oh["count"]
    validation_size = validation_fraction
    train_size = int(len(X) * (1 - validation_size))
    X_train, X_validation = X[0:train_size], X[train_size : len(X)]
    Y_train, Y_validation = y[0:train_size], y[train_size : len(X)]

    return X_train, X_validation, Y_train, Y_validation


def train_and_save(
    data, filename="finalized_model_24052022.sav", print_diagnostics=False
):
    X_train, X_validation, Y_train, Y_validation = split_train_test(
        data, validation_fraction=0.25
    )
    model = RandomForestRegressor(n_estimators=300)  # rbf is default kernel
    model.fit(X_train, Y_train)
    # save the model to disk
    dump(model, open(filename, "wb"))

    if print_diagnostics:
        predictions = model.predict(X_validation)
        result = mean_squared_error(Y_validation, predictions)
        print(result)


def load_model(filename="finalized_model_24052022.sav"):
    model = load(open(filename, "rb"))
    return model


def find_best_model(
    data, num_folds=5, scoring="neg_mean_squared_error"
):
    """
    Takes the one-hot encoded input and searches for the best model.
    Stores the result in a plot.
    """
    X_train, _, Y_train, _ = split_train_test(data, validation_fraction=0.25)
    # spot check the algorithms
    models = []
    models.append(("LR", LinearRegression()))
    models.append(("LASSO", Lasso()))
    models.append(("EN", ElasticNet()))
    models.append(("KNN", KNeighborsRegressor()))
    models.append(("CART", DecisionTreeRegressor()))
    models.append(("SVR", SVR()))
    # Neural Network
    models.append(("MLP", MLPRegressor()))
    # Ensable Models
    # Boosting methods
    models.append(("ABR", AdaBoostRegressor()))
    models.append(("GBR", GradientBoostingRegressor()))
    # Bagging methods
    models.append(("RFR", RandomForestRegressor()))
    models.append(("ETR", ExtraTreesRegressor()))

    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds)
        # converted mean square error to positive. The lower the beter
        cv_results = -1 * cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring=scoring
        )
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # compare algorithms
    fig = plt.figure()
    fig.suptitle("Algorithm Comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    fig.set_size_inches(15, 8)
    plt.savefig("plots/algorithm_comparison.png")


def optimize_RFR(data, num_folds=5, scoring="neg_mean_squared_error"):
    """
    Grid search for RandomForestRegressor hyperparameter and prints the result
    n_estimators : default=10 (The number of trees in the forest.)
    """
    X_train, _, Y_train, _ = split_train_test(data, validation_fraction=0.25)
    param_grid = {"n_estimators": [50, 100, 150, 200, 250, 300, 350, 400]}
    model = RandomForestRegressor()
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(
        estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold
    )
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def make_EDA_plots(data):
    """
    Create main explanatory data analysis plots
    """
    df = data
    df = prepare_data_for_EDA(data=df)
    check_plots_path()
    point_plotter(
        data=df,
        x="hour",
        y="count",
        hue="weekday",
        title="Hourly counts of rented bikes during weekdays and weekends",
    )
    point_plotter(
        data=df,
        x="hour",
        y="casual",
        hue="weekday",
        title="Hourly counts of rented bikes during weekdays and weekends for Unregistered users",
    )
    point_plotter(
        data=df,
        x="hour",
        y="registered",
        hue="weekday",
        title="Hourly counts of rented bikes during weekdays and weekends for Registered users",
    )
    point_plotter(
        data=df,
        x="hour",
        y="count",
        hue="weather",
        title="Hourly counts of rented bikes during different weathers conditions",
    )
    point_plotter(
        data=df,
        x="hour",
        y="count",
        hue="season",
        title="Hourly counts of rented bikes during different seasons",
    )
    bar_plotter(
        data=df,
        x="month",
        y="count",
        title="Hourly counts of rented bikes during during different months",
    )
    bar_plotter(
        data=df,
        x="weekday",
        y="count",
        title="Hourly counts of rented bikes during during different week days",
    )

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))
    sns.regplot(x=df["temp"], y=df["count"], ax=ax1)
    ax1.set(title="Relation between temperature and users")
    sns.regplot(x=df["humidity"], y=df["count"], ax=ax2)
    ax2.set(title="Relation between humidity and users")
    fig.savefig("plots/relation_humidity_and_temperature.png")

    corr = df.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr, annot=True, annot_kws={"size": 15})
    plt.savefig("plots/correlation_matrix.png")


def diagnose_scaling(data):
    """
    Create a plot to compare different scaling for the count parameter
    """
    df = data
    df = prepare_data_for_EDA(data=df)
    check_plots_path()

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(
        ncols=2, nrows=5, figsize=(20, 20)
    )
    sns.distplot(df["count"], ax=ax1)
    ax1.set(title="Distribution of the hourly count without any scaling")
    qqplot(df["count"], ax=ax2, line="s")
    ax2.set(title="Theoretical quantiles")

    df["count_log"] = np.log(df["count"])
    sns.distplot(df["count_log"], ax=ax3)
    ax3.set(title="Distribution of the hourly count with logarithmic scaling")
    qqplot(df["count_log"], ax=ax4, line="s")
    ax4.set(title="Theoretical quantiles")

    X = df[["count"]]
    transformer = MinMaxScaler().fit(X)
    df["count_normalized"] = transformer.transform(X)
    sns.distplot(df["count_normalized"], ax=ax5)
    ax5.set(title="Distribution of the hourly count with MinMax scaling")
    qqplot(df["count_normalized"], ax=ax6, line="s")
    ax6.set(title="Theoretical quantiles")

    transformer = PowerTransformer().fit(X)
    df["count_normalized"] = transformer.transform(X)
    sns.distplot(df["count_normalized"], ax=ax7)
    ax7.set(title="Distribution of the hourly count with PowerTransformer scaling")
    qqplot(df["count_normalized"], ax=ax8, line="s")
    ax8.set(title="Theoretical quantiles")

    transformer = QuantileTransformer().fit(X)
    df["count_normalized"] = transformer.transform(X)
    sns.distplot(df["count_normalized"], ax=ax9)
    ax9.set(title="Distribution of the hourly count with QuantileTransformer scaling")
    qqplot(df["count_normalized"], ax=ax10, line="s")
    ax10.set(title="Theoretical quantiles")

    fig.tight_layout()
    fig.savefig("plots/scaling_diagnostics.png")


def one_hot_encoding(data, column):
    """
    Convert the categorical data to encodings.
    """
    data = pd.concat(
        [data, pd.get_dummies(data[column], prefix=column, drop_first=True)], axis=1
    )
    data = data.drop([column], axis=1)
    return data


def check_plots_path():
    """
    Check if plots path exists and create it otherwise.
    """
    path = "./plots"
    isExist = os.path.exists(path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("Plots directory is created!")


def point_plotter(data, x, y, hue, title):
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.pointplot(data=data, x=x, y=y, hue=hue, ax=ax)
    ax.set(title=title)
    fig.savefig(f"plots/{title}.png")


def bar_plotter(data, x, y, title):
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.barplot(data=data, x=x, y=y, ax=ax)
    ax.set(title=title)
    fig.savefig(f"plots/{title}.png")


if __name__ == "__main__":
    df_ = load_data(print_diagnostics=True)
    # make_EDA_plots(data=df)
    # diagnose_scaling(data=df)
    df_oh_ = prepare_data(data=df_, print_diagnostics=True)
    # find_best_model(data=df_oh)
    # optimize_RFR(data=df_oh)
    train_and_save(data=df_oh_, print_diagnostics=True)
