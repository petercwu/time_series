import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Parse dates makes the column a datetime object instead of strings or ints
df = pd.read_csv("CTA_-_Ridership_-_Daily_Boarding_Totals.csv", parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]  # shorter names
# Sorts the dates and makes them the indices when accessing
# For example, df.loc['2023-06-30'] will retrieve the row(s)
# corresponding to the date '2023-06-30'
# df['2023-06-01':'2023-06-15'] will retrieve rows within the specified date range
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)
df = df.drop_duplicates()
print(df.head())

# Show an example of the ridership
df["2019-03":"2019-05"].plot(grid=True, marker=".", figsize=(8, 3.5))
plt.show()

# Visualize naive forecast by overlaying two time series
# as well as same time series lagged by one week since we see a weekly seasonality
# also plot the difference between the two

# .diff calculates difference between consecutive elements
# takes the values and the element 7 positions before and finds the difference
diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8,5))
df.plot(ax=axs[0], legend=False, marker=".") # Original time series
df.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":") # Lagged
diff_7.plot(ax=axs[1], grid=True, marker=".") # 7-day difference time series
plt.show()

print(diff_7.abs().mean())
# Naive forecast mean absolute error (MAE)
# bus     43915.608696
# rail    42143.271739

targets = df[["bus", "rail"]]["2019-03":"2019-05"]
MAPE = (diff_7 / targets).abs().mean()
print(MAPE)
# Mean absolute percentage error (MAPE)
# bus     0.082938
# rail    0.089948

# Check for yearly seasonality
period = slice("2001", "2019")
df_monthly = df.drop("day_type", axis=1)
df_monthly = df_monthly.resample("M").mean() # Computes the mean for each month
rolling_average_12_months = df_monthly[period].rolling(window=12).mean()

fig, ax = plt.subplots(figsize=(8, 4))
df_monthly[period].plot(ax=ax, marker=".")
rolling_average_12_months.plot(ax=ax, grid=True, legend=False)
plt.show()

# Plot 12 month differencing
# Removes the yearly seasonality and linear downward trend
df_monthly.diff(12)[period].plot(grid=True, marker=".", figsize=(8,3))
plt.show()

# Use ARIMA to predict the next day rail ridership
origin, today = "2019-01-01", "2019-05-31"
# asfreq("D") sets time series' frequency to daily
rail_series = df.loc[origin:today]["rail"].asfreq("D")
model = ARIMA(
    rail_series,
    order=(1, 0, 0),
    seasonal_order=(0, 1, 1, 7)
)
model = model.fit()
y_pred = model.forecast()
print(y_pred)
# Outputs
# 2019-06-01    427758.626286

# Use a loop to compute MAE over a period
origin, state_date, end_date = "2019-01-01", "2019-03-01", "2019-05-31"
time_period = pd.date_range(state_date, end_date)
rail_series = df.loc[origin:end_date]["rail"].asfreq("D")
y_preds = []
# .shift(-1) shifts the day one day earlier
# "today" starts feb 28 and predicts tomorrow which is the start date
for today in time_period.shift(-1):
    model = ARIMA(
        rail_series[origin:today],
        order=(1, 0, 0),
        seasonal_order=(0, 1, 1, 7),
    )
    model = model.fit()
    # ARIMA model returns a tuple containing
    # the forecasted values, prediction standard errors, and confidence intervals
    # [0] access only forecasted values
    y_pred = model.forecast()[0]
    y_preds.append(y_pred)

# Make y-preds a series so you can subtract the rail_series from it to get MAE
y_preds = pd.Series(y_preds, index=time_period)
mae = (y_preds - rail_series[time_period]).abs().mean()
print(mae)
# Outputs
# 32040.720092085743