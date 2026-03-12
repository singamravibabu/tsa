# Time Series Analysis — Streamlit Interactive Dashboard

## Project Structure

```
time_series_dashboard/
│
├── app.py
├── data_generator.py
├── analysis.py
├── requirements.txt
```

---

# 1️⃣ Install Libraries

`requirements.txt`

```txt
streamlit
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

Run app:

```bash
streamlit run app.py
```

---

# 2️⃣ data_generator.py

Creates synthetic time-series data with **trend + seasonality + noise**.

```python
import pandas as pd
import numpy as np


def generate_data():

    date_rng = pd.date_range(
        start="2015-01-01",
        end="2023-12-31",
        freq="M"
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

# 3️⃣ analysis.py

Contains functions used in the dashboard.

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def moving_average(df, window):

    return df["Sales"].rolling(window=window).mean()


def exponential_smoothing(df, alpha):

    return df["Sales"].ewm(alpha=alpha, adjust=False).mean()


def first_difference(df):

    return df["Sales"].diff()


def seasonal_difference(df, lag):

    return df["Sales"].diff(lag)


def adf_test(series):

    result = adfuller(series.dropna())

    return {
        "ADF Statistic": result[0],
        "p-value": result[1]
    }
```

---

# 4️⃣ Streamlit Dashboard (app.py)

```python
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from data_generator import generate_data
from analysis import (
    moving_average,
    exponential_smoothing,
    first_difference,
    seasonal_difference,
    adf_test
)

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


st.title("📊 Time Series Analysis Dashboard")

st.write("Interactive exploration of time series data.")


df = generate_data()

st.sidebar.header("Controls")


window = st.sidebar.slider(
    "Moving Average Window",
    2, 24, 12
)

alpha = st.sidebar.slider(
    "Exponential Smoothing Alpha",
    0.1, 0.9, 0.3
)

lag = st.sidebar.slider(
    "Seasonal Difference Lag",
    1, 24, 12
)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Time Series",
    "Smoothing",
    "Decomposition",
    "Stationarity",
    "Correlation"
])


# ---------- TAB 1 ----------
with tab1:

    st.subheader("Original Time Series")

    fig, ax = plt.subplots()

    sns.lineplot(x=df.index, y=df["Sales"], ax=ax)

    ax.set_title("Sales Over Time")

    st.pyplot(fig)


# ---------- TAB 2 ----------
with tab2:

    st.subheader("Smoothing Techniques")

    df["SMA"] = moving_average(df, window)

    df["EMA"] = exponential_smoothing(df, alpha)

    fig, ax = plt.subplots()

    sns.lineplot(x=df.index, y=df["Sales"], label="Original", ax=ax)

    sns.lineplot(x=df.index, y=df["SMA"], label="SMA", ax=ax)

    sns.lineplot(x=df.index, y=df["EMA"], label="EMA", ax=ax)

    ax.set_title("Smoothing Comparison")

    st.pyplot(fig)


# ---------- TAB 3 ----------
with tab3:

    st.subheader("Time Series Decomposition")

    decomposition = seasonal_decompose(
        df["Sales"],
        model="additive",
        period=12
    )

    fig = decomposition.plot()

    st.pyplot(fig)


# ---------- TAB 4 ----------
with tab4:

    st.subheader("Stationarity Check")

    df["FirstDiff"] = first_difference(df)

    fig, ax = plt.subplots()

    sns.lineplot(
        x=df.index,
        y=df["FirstDiff"],
        ax=ax
    )

    ax.set_title("First Difference")

    st.pyplot(fig)

    st.subheader("ADF Test")

    result = adf_test(df["Sales"])

    st.write(result)


# ---------- TAB 5 ----------
with tab5:

    st.subheader("Autocorrelation")

    fig1, ax1 = plt.subplots()

    plot_acf(df["Sales"], ax=ax1)

    st.pyplot(fig1)

    st.subheader("Partial Autocorrelation")

    fig2, ax2 = plt.subplots()

    plot_pacf(df["Sales"], ax=ax2)

    st.pyplot(fig2)
```

---

# Dashboard Features

Students can interactively explore:

### Time Series Visualization

* Line plots

### Smoothing

* Moving Average window
* Exponential smoothing α

### Decomposition

* Trend
* Seasonal
* Residual

### Stationarity

* Differencing
* ADF Test

### Correlation

* ACF
* PACF

All these correspond to the **lessons in the uploaded PPT**. 

---

# What the Dashboard Looks Like

```
-----------------------------------
Time Series Analysis Dashboard
-----------------------------------

Sidebar Controls
   Moving Average Window
   Alpha
   Seasonal Lag

Tabs
  Time Series
  Smoothing
  Decomposition
  Stationarity
  Correlation

Interactive Charts
-----------------------------------
