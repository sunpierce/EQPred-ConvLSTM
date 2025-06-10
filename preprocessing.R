library(zoo)
library(dplyr)
library(lubridate)
library(imputeTS)
library(ggplot2)
library(tidyr)
library(scales)
library(readr)

#### Groundwater ####
# Load data
groundwater = read.csv("~/Desktop/Academia Sinica/EQPred-ConvLSTM/dataset/Groundwater.csv", check.names = F)
station_cols = colnames(groundwater)[5:ncol(groundwater)]

# Step 1: Linear interpolation for each station
for (col in station_cols) {
  groundwater[[col]] = na_kalman(groundwater[[col]], model = "auto.arima")
}

# Step 2: Demeaning (subtract station-wise means)
station_means = sapply(groundwater[station_cols], mean)
for (col in station_cols) {
  groundwater[[col]] = groundwater[[col]] - station_means[[col]]
}

# Step 3: Seasonal differencing (lag = 365 * 24 hours)
lag_hours = 365 * 24
for (col in station_cols) {
  groundwater[[col]] = groundwater[[col]] - dplyr::lag(groundwater[[col]], lag_hours)
}

# Remove NA induced by differencing
groundwater = groundwater[(lag_hours+1):nrow(groundwater),]

# Step 4: Average by one day
groundwater = groundwater %>%
  mutate(Date = make_date(Year, Month, Day))

daily_groundwater = groundwater %>%
  group_by(Date) %>%
  summarise(across(where(is.numeric), mean, na.rm = F)) %>%
  ungroup()

daily_groundwater = daily_groundwater[,c(2:4,6:ncol(daily_groundwater))]

# Done — groundwater now contains the preprocessed time series
write_csv(groundwater, "~/Desktop/Academia Sinica/EQPred-ConvLSTM/dataset/preprocessed_groundwater.csv")
write_csv(daily_groundwater, "~/Desktop/Academia Sinica/EQPred-ConvLSTM/dataset/daily_groundwater.csv")

#### Figure Demo ####

station_id = "11150211"
original_series = original[, 15]
imputed_series = imputed[, 15]
processed_series = groundwater[, 15]

# Create time index
time_index = seq.POSIXt(from = as.POSIXct("2008-04-01 00:00", tz = "UTC"),
                        by = "hour", length.out = length(original_series))

# Identify imputed values (NA in raw)
imputed_idx = which(is.na(original_series))

# Combine into data frame
df_plot = data.frame(
  Time = time_index,
  Raw = original_series,
  Kalman = imputed_series,
  Final = processed_series
)

# Convert to long format
df_long = df_plot %>%
  pivot_longer(cols = -Time, names_to = "Stage", values_to = "Value")

# Flag imputed points in Kalman
df_long$Imputed = FALSE
df_long$Imputed[df_long$Stage == "Kalman" & df_long$Time %in% time_index[imputed_idx]] = TRUE

# Factor to control plot order
df_long$Stage = factor(df_long$Stage, levels = c("Raw", "Kalman", "Final"))

# Plot
ggplot(df_long, aes(x = Time, y = Value)) +
  geom_line(color = "black", linewidth = 0.4) +
  geom_point(data = subset(df_long, Stage == "Kalman" & Imputed == TRUE),
             aes(x = Time, y = Value),
             color = "red", size = 0.5) +
  facet_wrap(~Stage, ncol = 1, scales = "free_y") +
  labs(x = "Time", y = "Groundwater Level (m)") +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(size = 12, face = "bold")
  )

#### Earthquake ####
# Read earthquake data
earthquake = read_csv("~/Desktop/Academia Sinica/EQPred-ConvLSTM/dataset/Earthquake.csv") %>%
  mutate(Date = as.Date(sprintf("%04d-%02d-%02d", Year, Month, Day)),
         Occur = 1)

# Create full date sequence
date_seq = seq.Date(from = as.Date("2009-02-01"), to = as.Date("2023-03-31"), by = "day")
full_dates = data.frame(Date = date_seq)

# Identify dates with no earthquakes
no_eq_dates = anti_join(full_dates, earthquake, by = "Date") %>%
  mutate(Year = year(Date),
         Month = month(Date),
         Day = day(Date),
         Occur = 0) %>%
  select(Year, Month, Day, Occur)

# Prepare original earthquake data (keep duplicates)
eq_with_occur = earthquake %>%
  select(Year, Month, Day, Occur)

# Combine both
final_dataset = bind_rows(eq_with_occur, no_eq_dates) %>%
  arrange(Year, Month, Day)

write_csv(final_dataset, "~/Desktop/Academia Sinica/EQPred-ConvLSTM/dataset/Earthquake_Occurrence.csv")
