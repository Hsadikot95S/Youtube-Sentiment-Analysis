# Import libraries and files
import pandas as p
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

# Create a dataframe from video statistics
def performPredictions(channelName):
    dataframe = p.read_json("comments/" + channelName + "_stats.json")
    dataframe.info()
    dataframe.shape
    """
    Data Frame Created

    Data columns (total 25 columns):
    #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
    0   kind              25 non-null     object 
    1   etag              25 non-null     object 
    2   id                25 non-null     object 
    3   title             25 non-null     object 
    4   positive_vader    25 non-null     float64
    5   negative_vader    25 non-null     float64
    6   neutral_vader     25 non-null     int64  
    7   positive_afinn    25 non-null     float64
    8   negative_afinn    25 non-null     int64  
    9   neutral_afinn     25 non-null     float64
    10  anger_NRC         25 non-null     float64
    11  anticipation_NRC  25 non-null     float64
    12  disgust_NRC       25 non-null     float64
    13  fear_NRC          25 non-null     float64
    14  joy_NRC           25 non-null     float64
    15  negative_NRC      25 non-null     float64
    16  positive_NRC      25 non-null     float64
    17  sadness_NRC       25 non-null     float64
    18  surprise_NRC      25 non-null     float64
    19  trust_NRC         25 non-null     float64
    20  viewCount         25 non-null     int64  
    21  likeCount         25 non-null     int64  
    22  dislikeCount      25 non-null     int64  
    23  commentCount      25 non-null     int64  
    24  likedislikeratio  25 non-null     float64

    """
    vader_prediction(dataframe)
    afinn_prediction(dataframe)
    nrc_prediction(dataframe)


# Predict like/dislike ratio using Vader polarity score
def vader_prediction(dataframe):
    print("########################## Vader ##########################")
    X = p.DataFrame(dataframe, columns=["positive_vader", "negative_vader", "neutral_vader", "viewCount", "commentCount"])
    #

    X.head()
    Y = p.DataFrame(dataframe, columns=["likedislikeratio"])
    # split train test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    # y_test = sc_y.fit_transform(y_test)

    # train Linear Regression Model
    LinR = LinearRegression()

    fitResult = LinR.fit(X_train, y_train)
    y_pred = fitResult.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))
    # train MLP Regressor Model
    mlp = MLPRegressor(random_state=0, activation="relu", hidden_layer_sizes=16)

    mlp.fit(X_train, y_train)

    predictedValues = mlp.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, predictedValues))
    print("MSE:", mean_squared_error(y_test, predictedValues))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictedValues)))
    print("R2:", r2_score(y_test, predictedValues))

    # define base model
    def baseline_model_vader():
        model = Sequential()
        model.add(Dense(12, input_dim=5, kernel_initializer="normal", activation="relu"))
        model.add(Dense(1, activation="relu"))
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
        return model

    # evaluate model
    estimators = []
    estimators.append(("standardize", StandardScaler()))
    estimators.append(("mlp", KerasRegressor(build_fn=baseline_model_vader, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10) if len(dataframe.index) > 10 else KFold(n_splits=len(dataframe.index))
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    print("RMSE: %2.f" % (results.std() ** (1 / 2)))


# Predict like/dislike ratio using Afinn polarity score
def afinn_prediction(dataframe):
    print("########################## Afinn ##########################")

    X = p.DataFrame(dataframe, columns=["positive_afinn", "negative_afinn", "neutral_afinn", "viewCount", "commentCount"])
    #

    X.head()

    # Y = dataframe['likedislikeratio']
    Y = p.DataFrame(dataframe, columns=["likedislikeratio"])
    # split train test data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    # y_test = sc_y.fit_transform(y_test)

    # train Linear regression Model
    LinR = LinearRegression()

    fitResult = LinR.fit(X_train, y_train)
    y_pred = fitResult.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))

    # train MLP Regressor Model

    mlp = MLPRegressor(random_state=0, activation="relu", hidden_layer_sizes=16)

    mlp.fit(X_train, y_train)

    predictedValues = mlp.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, predictedValues))
    print("MSE:", mean_squared_error(y_test, predictedValues))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictedValues)))
    print("R2:", r2_score(y_test, predictedValues))

    # load dataset
    # define base model
    def baseline_model_afinn():
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=5, kernel_initializer="normal", activation="relu"))
        model.add(Dense(1, activation="relu"))
        # Compile model
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
        return model

    # evaluate model
    estimators = []
    estimators.append(("standardize", StandardScaler()))
    estimators.append(("mlp", KerasRegressor(build_fn=baseline_model_afinn, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10) if len(dataframe.index) > 10 else KFold(n_splits=len(dataframe.index))
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    print("RMSE: %2.f" % (results.std() ** (1 / 2)))


# Predict like/dislike ratio using NRC Lexicon sentiment score
def nrc_prediction(dataframe):
    print("########################## NRC ##########################")

    X = p.DataFrame(
        dataframe,
        columns=[
            "anger_NRC",
            "anticipation_NRC",
            "disgust_NRC",
            "fear_NRC",
            "joy_NRC",
            "negative_NRC",
            "positive_NRC",
            "sadness_NRC",
            "surprise_NRC",
            "trust_NRC",
            "viewCount",
            "commentCount",
        ],
    )
    #

    X.head()

    # Y = dataframe['likedislikeratio']
    Y = p.DataFrame(dataframe, columns=["likedislikeratio"])
    # split train test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    # y_test = sc_y.fit_transform(y_test)

    # train Linear Regression Model
    LinR = LinearRegression()

    fitResult = LinR.fit(X_train, y_train)
    y_pred = fitResult.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))

    # train MLP Regressor Model
    mlp = MLPRegressor(random_state=0, activation="relu", hidden_layer_sizes=16)

    mlp.fit(X_train, y_train)

    predictedValues = mlp.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, predictedValues))
    print("MSE:", mean_squared_error(y_test, predictedValues))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictedValues)))
    print("R2:", r2_score(y_test, predictedValues))

    # load dataset
    # define base model
    def baseline_model_NRC():
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=12, kernel_initializer="normal", activation="relu"))
        model.add(Dense(1, activation="relu"))
        # Compile model
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
        return model

    # evaluate model
    estimators = []
    estimators.append(("standardize", StandardScaler()))
    estimators.append(("mlp", KerasRegressor(build_fn=baseline_model_NRC, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10) if len(dataframe.index) > 10 else KFold(n_splits=len(dataframe.index))
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    print("RMSE: %2.f" % (results.std() ** (1 / 2)))
