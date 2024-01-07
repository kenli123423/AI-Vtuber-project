import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Import part
df = pd.read_csv("F:\Data for py\pyrsi.csv")
slr = LinearRegression()
df = df.iloc[20:]
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df = df.sort_values(by='Date')
y = df['Close']
X = df[['20MA']]


# Adjustment and sorting of date

def linear_pred(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    slr.fit(X_train, y_train)
    print("The coefficient of this line is",slr.coef_)
    print("The y-intercept of this line is",slr.intercept_)
    y_train_pre = slr.predict(X_train)
    y_test_pre = slr.predict(X_test)
    plt.scatter(X_train, y_train, c='steelblue', marker='o', edgecolors='white', label='Training data')
    plt.scatter(X_test, y_test, c='red', marker='s', edgecolors='white', label='Test data')
    plt.xlabel("20 moving average (US dollars)")
    plt.ylabel("Stock price(US dollars)")
    plt.title("Correlation between 20ma and stock price")
    plt.legend(loc='upper right')
    plt.plot(X_train.values, y_train_pre, color='black', lw=2)
    plt.show()

    def performance_test(y_train, y_train_pre, y_test, y_test_pre):
        from sklearn.metrics import mean_squared_error
        r2 = r2_score(y_test, y_test_pre)
        mse = mean_squared_error(y_test, y_test_pre)
        print("MSE:", mse)
        print("r2_score:",r2)

        def res(y_train, y_train_pre, y_test, y_test_pre):
            plt.scatter(y_train_pre, abs(y_train - y_train_pre), c='steelblue', label='Training data')
            plt.scatter(y_test_pre, abs(y_test - y_test_pre), c='red', label="Test data")
            plt.title("Correlation between predicted values and residuals")
            plt.xlabel("Predicted values")
            plt.ylabel("Residuals")
            plt.legend(loc='upper right')
            plt.show()
            plt.hist(y_train - y_train_pre, bins=30)
            plt.title("Frequency of residuals value")
            plt.xlabel("Residuals value")
            plt.ylabel("No. of residuals")
            plt.show()
        res(y_train, y_train_pre, y_test, y_test_pre)

    performance_test(y_train, y_train_pre, y_test, y_test_pre)


linear_pred(X, y)
