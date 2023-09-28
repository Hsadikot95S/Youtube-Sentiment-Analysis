# Import libraries and files
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

# LSTM model to predict future sentiment using Time series data
def LongShortTermMemory(dataframe_LSTM, algoName, channelName, algo):
    dataframe_LSTM.index = dataframe_LSTM.date
    dataframe_LSTM.drop("date", axis=1, inplace=True)
    dataset = dataframe_LSTM.values
    split = int(0.7 * len(dataset))
    train = dataset[0:split, :]
    valid = dataset[split + 1 :, :]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []
    for i in range(int(0.3 * (len(train))), len(train)):
        x_train.append(scaled_data[i - int(0.3 * (len(train))) : i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=100))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="Adagrad")
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
    inputs = dataframe_LSTM[len(dataframe_LSTM) - len(valid) - int(0.3 * (len(train))) :].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(int(0.3 * (len(train))), inputs.shape[0]):
        X_test.append(inputs[i - int(0.3 * (len(train))) : i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predict = model.predict(X_test)
    predict = scaler.inverse_transform(predict)
    # Calculate RMSE Value
    rms = np.sqrt(np.mean(np.power((valid - predict), 2)))
    print(rms)
    train = dataframe_LSTM[:split]
    valid = dataframe_LSTM[split + 1 :]
    valid["Predictions"] = predict
    # Visualization of result
    plt.plot(train[algoName])
    plt.plot(valid[[algoName, "Predictions"]])
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    plt.suptitle("LSTM Sentiment Predictions for " + channelName + " Using" + algo, fontsize=30)
    # plt.title()
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.savefig("images/" + channelName + "_LSTM_" + algo + ".png", dpi=600)
    plt.show()


# Linear and Polynomial regression Model to predict future sentiment using Time Series data
def Linear_Poly_Regression(dates_df, dataframe, algoName, channelName, algo):
    # Store the original dates for plotting the predicitons
    dates_df["date"] = dates_df["date"].astype(str)

    dates_df["date"] = dates_df["date"].str.replace("\D", "").astype(int)
    org_dates = dataframe["date"]
    dates_df.tail()

    polarities = dates_df[algoName].to_numpy()
    dates = dates_df["date"].to_numpy()

    # Convert to 1d Vector
    dates = np.reshape(dates, (len(dates), 1))
    polarities = np.reshape(polarities, (len(polarities), 1))

    org_dates = org_dates.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(dates, polarities, test_size=0.33, random_state=42)
    X_train

    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train, y_train)
    clfpoly2 = make_pipeline(PolynomialFeatures(4), Ridge())
    clfpoly2.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    poly_pred = clfpoly2.predict(X_test)
    # Caluclate RMSE value
    rmse_lr = np.sqrt(np.mean(np.power((np.transpose(np.array(y_test)) - np.array(lr_pred)), 2)))
    rmse_pr = np.sqrt(np.mean(np.power((np.transpose(np.array(y_test)) - np.array(poly_pred)), 2)))

    print(rmse_lr, rmse_pr)
    # Linear Zoomed
    # plt.figure(figsize=(12, 6))
    # plt.plot(org_dates[len(X_train) :], polarities[len(X_train) :], color="black", label="Data")
    # plt.plot(org_dates[len(X_train) :], lr_pred, color="red", label="Linear Regression")
    # plt.xlabel("Date")
    # plt.ylabel("Polarities")
    # plt.legend()
    # ax = plt.axes()
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # plt.suptitle("Linear Regression Sentiment Predictions for " + channelName + " Using" + algo, fontsize=25)
    # plt.savefig("images/" + channelName + "_Linear_Zoomed_" + algoName + ".png")
    # plt.show()
    # Linear
    plt.figure(figsize=(12, 6))
    plt.plot(org_dates, polarities, color="black", label="Data")
    plt.plot(org_dates[len(X_train) :], lr_pred, color="red", label="Linear Regression")
    plt.xlabel("Date")
    plt.ylabel("Polarities")
    plt.legend()
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.suptitle("Linear Regression Sentiment Predictions for " + channelName + " Using" + algo, fontsize=25)
    plt.savefig("images/" + channelName + "_Linear_" + algo + ".png")
    plt.show()
    # Polynomial Zommed
    # plt.figure(figsize=(12, 6))
    # plt.plot(org_dates[len(X_train) :], polarities[len(X_train) :], color="black", label="Data")
    # plt.plot(org_dates[len(X_train) :], poly_pred, color="blue", label="Polynomial Regression")
    # plt.xlabel("Date")
    # plt.ylabel("Polarities")
    # plt.legend()
    # plt.savefig("images/" + channelName + "_Polynomial_Zoomed_" + algoName + ".png")
    # plt.show()
    # Polynomial
    plt.figure(figsize=(12, 6))
    plt.plot(org_dates, polarities, color="black", label="Data")
    plt.plot(org_dates[len(X_train) :], poly_pred, color="blue", label="Polynomial Regression")
    plt.xlabel("Date")
    plt.ylabel("Polarities")
    plt.legend()
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.suptitle("Polynomial Regression Sentiment Predictions for " + channelName + " Using" + algo, fontsize=25)
    plt.savefig("images/" + channelName + "_Polynomial_" + algo + ".png")
    plt.show()


# Driver function
def performPredictions(data, channelName):
    dataframe = pd.DataFrame(data, columns=["date", "polarity_vader_avg", "no_comments", "afinn_score_avg"])
    LongShortTermMemory(dataframe[["date", "polarity_vader_avg"]], "polarity_vader_avg", channelName, "Vader")
    LongShortTermMemory(dataframe[["date", "afinn_score_avg"]], "afinn_score_avg", channelName, "Afinn")
    Linear_Poly_Regression(dataframe[["date", "polarity_vader_avg"]], dataframe, "polarity_vader_avg", channelName, "Vader")
    Linear_Poly_Regression(dataframe[["date", "afinn_score_avg"]], dataframe, "afinn_score_avg", channelName, "Afinn")
