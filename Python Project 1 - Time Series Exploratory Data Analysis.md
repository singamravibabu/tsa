# Python Project: Time Series Exploratory Data Analysis

## Project Folder Structure

```
time_series_project/
│
├── main.py
├── data_generator.py
├── visualization.py
├── smoothing.py
├── decomposition.py
├── stationarity.py
├── differencing.py
├── correlation_analysis.py
└── requirements.txt
```

---

# 1️⃣ requirements.txt

```txt
pandas
numpy
matplotlib
seaborn
statsmodels
```

Install:

```bash
pip install -r requirements.txt
```

---

# 2️⃣ data_generator.py

Generate synthetic time series data (trend + seasonality + noise).

```python
import pandas as pd
import numpy as np


def generate_sales_data():
    
    date_rng = pd.date_range(
        start='2015-01-01',
        end='2023-12-31',
        freq='M'
    )

    np.random.seed(42)

    trend = np.linspace(100, 300, len(date_rng))

    seasonality = 20 * np.sin(
        2 * np.pi * date_rng.month / 12
    )

    noise = np.random.normal(0, 8, len(date_rng))

    sales = trend + seasonality + noise

    df = pd.DataFrame({
        "Date": date_rng,
        "Sales": sales
    })

    df.set_index("Date", inplace=True)

    return df
```

---

# 3️⃣ visualization.py

Line plot + Box plots.

```python
import matplotlib.pyplot as plt
import seaborn as sns


def line_plot(df):

    plt.figure(figsize=(12,6))

    sns.lineplot(x=df.index, y=df["Sales"])

    plt.title("Monthly Sales Over Time")

    plt.grid(True)

    plt.show()


def monthly_box_plot(df):

    df["Month"] = df.index.month

    plt.figure(figsize=(10,6))

    sns.boxplot(x="Month", y="Sales", data=df)

    plt.title("Sales Distribution by Month")

    plt.show()


def yearly_box_plot(df):

    df["Year"] = df.index.year

    plt.figure(figsize=(10,6))

    sns.boxplot(x="Year", y="Sales", data=df)

    plt.title("Sales Distribution by Year")

    plt.show()
```

---

# 4️⃣ smoothing.py

Moving averages + exponential smoothing.

```python
import seaborn as sns
import matplotlib.pyplot as plt


def moving_average(df):

    df["SMA_3"] = df["Sales"].rolling(window=3).mean()

    df["SMA_6"] = df["Sales"].rolling(window=6).mean()

    df["SMA_12"] = df["Sales"].rolling(window=12).mean()

    sns.lineplot(x=df.index, y=df["Sales"], label="Original", alpha=0.4)

    sns.lineplot(x=df.index, y=df["SMA_3"], label="3-Month SMA")

    sns.lineplot(x=df.index, y=df["SMA_6"], label="6-Month SMA")

    sns.lineplot(x=df.index, y=df["SMA_12"], label="12-Month SMA")

    plt.title("Moving Average Comparison")

    plt.show()


def exponential_smoothing(df):

    df["ES_0.3"] = df["Sales"].ewm(alpha=0.3, adjust=False).mean()

    df["ES_0.5"] = df["Sales"].ewm(alpha=0.5, adjust=False).mean()

    sns.lineplot(x=df.index, y=df["Sales"], label="Original", alpha=0.4)

    sns.lineplot(x=df.index, y=df["ES_0.3"], label="α=0.3")

    sns.lineplot(x=df.index, y=df["ES_0.5"], label="α=0.5")

    plt.title("Exponential Smoothing")

    plt.show()
```

---

# 5️⃣ decomposition.py

Time series decomposition.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def perform_decomposition(df):

    decomposition = seasonal_decompose(
        df["Sales"],
        model="additive",
        period=12
    )

    decomposition.plot()

    plt.show()
```

---

# 6️⃣ stationarity.py

Stationarity check + ADF test.

```python
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def check_stationarity(df):

    roll_mean = df["Sales"].rolling(12).mean()

    roll_std = df["Sales"].rolling(12).std()

    plt.plot(df["Sales"], label="Original")

    plt.plot(roll_mean, label="Rolling Mean")

    plt.plot(roll_std, label="Rolling Std")

    plt.legend()

    plt.title("Rolling Statistics")

    plt.show()

    result = adfuller(df["Sales"])

    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
```

---

# 7️⃣ differencing.py

Remove trend and seasonality.

```python
import matplotlib.pyplot as plt


def first_difference(df):

    df["FirstDiff"] = df["Sales"].diff()

    plt.plot(df["FirstDiff"], color="red")

    plt.title("First Order Differencing")

    plt.show()


def seasonal_difference(df):

    df["SeasonalDiff"] = df["Sales"].diff(12)

    plt.plot(df["SeasonalDiff"], color="green")

    plt.title("Seasonal Differencing")

    plt.show()
```

---

# 8️⃣ correlation_analysis.py

ACF, PACF, lag plots.

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot


def autocorrelation(df):

    plot_acf(df["Sales"], lags=24)

    plt.show()


def partial_autocorrelation(df):

    plot_pacf(df["Sales"], lags=24)

    plt.show()


def lag_plot_visual(df):

    plt.figure(figsize=(5,5))

    lag_plot(df["Sales"])

    plt.show()
```

---

# 9️⃣ main.py

Main program running all lessons.

```python
from data_generator import generate_sales_data

from visualization import line_plot, monthly_box_plot, yearly_box_plot

from smoothing import moving_average, exponential_smoothing

from decomposition import perform_decomposition

from stationarity import check_stationarity

from differencing import first_difference, seasonal_difference

from correlation_analysis import (
    autocorrelation,
    partial_autocorrelation,
    lag_plot_visual
)


def main():

    df = generate_sales_data()

    line_plot(df)

    monthly_box_plot(df)

    yearly_box_plot(df)

    moving_average(df)

    exponential_smoothing(df)

    perform_decomposition(df)

    check_stationarity(df)

    first_difference(df)

    seasonal_difference(df)

    autocorrelation(df)

    partial_autocorrelation(df)

    lag_plot_visual(df)


if __name__ == "__main__":
    main()
```

---

# What This Project Teaches


| Lesson | Technique             |
| ------ | --------------------- |
| 1      | Line plots, box plots |
| 2      | Trend + seasonality   |
| 3      | Decomposition         |
| 4      | Moving averages       |
| 5      | Exponential smoothing |
| 6      | Stationarity tests    |
| 7      | Differencing          |
| 8      | ACF, PACF, lag plots  |

All these steps correspond directly to the material in the uploaded slides.
