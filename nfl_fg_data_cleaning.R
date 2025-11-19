# =============================================
# ----- NFL Data Exploration and Cleaning -----
# =============================================

# Data source: https://github.com/nflverse/nflfastR
# Goal: Predict whether a field goal is made.
# Approach:
#   1) Load play-by-play data (2001–2024)
#   2) Create Stadium Abbreviation and Game Date Variables
#   3) Create binary indoor/outdoor vairable
#   4) Impute missing weather (temp & wind)
#   5) Create binary precipitation variable
#   6) Clean field surface and create binary grass/turf indicator
#   7) Build final modeling dataset
#   8) Check for missing data
#   9) Explore data
#   10) Save CSV



# ================================================
# ----- 0) Clear Environment & Load Packages ----- 
# ================================================

rm(list = ls())

library(tidyverse)  
library(nflfastR)   
library(lubridate)  



# =========================================
# ----- 1) Load NFL Play-by-Play Data ----- 
# =========================================
# Data technically exists back to 1999, but weather
# is only consistent from 2001 onward. Also, 2025
# is still being played, so we stop at 2024.

seasons <- 2001:2024

pbp_all <- map_df(seasons, ~ load_pbp(.x))


# ===========================================================
# ----- 2) Stadium Abbreviation and Game Date Variables -----
# ===========================================================

# This will be used to help impute weather values in section 4

pbp_all <- pbp_all %>%
  mutate(
    stadium_abbr = str_extract(game_id, "[A-Z]{2,3}$"),
    game_date    = as.Date(game_date),
    game_month   = month(game_date)
  )



# ================================================
# ----- 3) Create Indoor / Outdoor Variable ------
# ================================================

# Valid roof values in nflfastR: "outdoors", "indoors", "dome", "closed", "open"
# If stadium is indoor, value is 1

pbp_all <- pbp_all %>%
  mutate(
    indoor = if_else(roof %in% c("dome", "closed"), 1, 0)
  )



# ================================================
# ----- 4) Impute Missing Weather (Temp/Wind) ----
# ================================================

# First, impute NA values from indoor games with 0 for wind and 70 for temp (the 
# average temperature for a climate controlled game)

# ----- 4a) Indoor games: fixed climate (70°F, 0 wind) -----
# ----------------------------------------------------------

pbp_all <- pbp_all %>%
  mutate(
    temp = ifelse(indoor == 1 & is.na(temp), 70, temp),
    wind = ifelse(indoor == 1 & is.na(wind), 0, wind)
  )


# Next, impute average values for missing outdoor games 
# While the most accurate way to impute wind and temperature is to calculate the 
# average wind and temperature for each stadium for each month, this doesn't fix 
# every case, so a tiered system will be used, from most specific to least specific


# ----- 4b) impute using stadium x month -----
# --------------------------------------------

# Outdoor stadium × month climatology
stadium_month_weather <- pbp_all %>%
  filter(indoor == 0) %>%
  group_by(stadium_abbr, game_month) %>%
  summarize(
    avg_temp = mean(temp, na.rm = TRUE),
    avg_wind = mean(wind, na.rm = TRUE),
    .groups = "drop"
  )

# Join climatology back and fill outdoor missing values
pbp_all <- pbp_all %>%
  left_join(stadium_month_weather,
            by = c("stadium_abbr", "game_month")) %>%
  mutate(
    temp = ifelse(indoor == 0 & is.na(temp), avg_temp, temp),
    wind = ifelse(indoor == 0 & is.na(wind), avg_wind, wind)
  ) %>%
  select(-avg_temp, -avg_wind)

# Check if missing data still exists
colSums(is.na(pbp_all[, c("temp", "wind")]))
# Since we still have NAs, move on to second tier


# ----- 4c) impute using stadium x month -----
# --------------------------------------------

stadium_state_map <- tibble::tribble(
  ~stadium_abbr, ~state,
  "ARI","AZ","ATL","GA","BAL","MD","BUF","NY","CAR","NC",
  "CHI","IL","CIN","OH","CLE","OH","DAL","TX","DEN","CO",
  "DET","MI","GB","WI","HOU","TX","IND","IN","JAX","FL",
  "KC","MO","LA","CA","LAC","CA","LV","NV","MIA","FL",
  "MIN","MN","NE","MA","NO","LA","NYG","NJ","NYJ","NJ",
  "PHI","PA","PIT","PA","SEA","WA","SF","CA","TB","FL",
  "TEN","TN","WAS","MD"
)

pbp_all <- pbp_all %>%
  left_join(stadium_state_map, by = "stadium_abbr")

state_month_weather <- pbp_all %>%
  filter(indoor == 0) %>%
  group_by(state, game_month) %>%
  summarize(
    avg_temp_state = mean(temp, na.rm = TRUE),
    avg_wind_state = mean(wind, na.rm = TRUE),
    .groups = "drop"
  )

pbp_all <- pbp_all %>%
  left_join(state_month_weather,
            by = c("state", "game_month")) %>%
  mutate(
    temp = ifelse(indoor == 0 & is.na(temp), avg_temp_state, temp),
    wind = ifelse(indoor == 0 & is.na(wind), avg_wind_state, wind)
  ) %>%
  select(-avg_temp_state, -avg_wind_state)

# Check if missing data still exists
colSums(is.na(pbp_all[, c("temp", "wind")]))
# Since we still have NAs, move on to second tier


# ----- 4d) impute using monthly averages only -----
# --------------------------------------------------

month_weather <- pbp_all %>%
  filter(indoor == 0) %>%
  group_by(game_month) %>%
  summarize(
    avg_temp_month = mean(temp, na.rm = TRUE),
    avg_wind_month = mean(wind, na.rm = TRUE),
    .groups = "drop"
  )

pbp_all <- pbp_all %>%
  left_join(month_weather, by = "game_month") %>%
  mutate(
    temp = ifelse(indoor == 0 & is.na(temp), avg_temp_month, temp),
    wind = ifelse(indoor == 0 & is.na(wind), avg_wind_month, wind)
  ) %>%
  select(-avg_temp_month, -avg_wind_month)

# Check if missing data still exists
colSums(is.na(pbp_all[, c("temp", "wind")]))
# There are no NAs, so we can move on now


# ===================================================
# ----- 5) Create Binary Precipitation Variable -----
# ===================================================

# To simplify the weather variable, we are creating a precipitation variable. 
# The code below is designed to identify if there are any indicators of precipitation
# in the weather description. If none are present, it will be set to (no precipitation)

pbp_all <- pbp_all %>%
  mutate(
    weather_low = tolower(trimws(weather)),
    
    precipitation = case_when(
      # Any precipitation keywords → 1
      str_detect(weather_low,
                 "rain|shower|drizzle|wet|snow|flurries|sleet|wintry|ice|storm|thunder") ~ 1,
      
      # Indoor stadiums → 0
      indoor == 1 ~ 0,
      
      # Has weather text but no precip words → 0
      !is.na(weather_low) & weather_low != "" ~ 0,
      
      # Everything else (missing or blank) → 0
      TRUE ~ 0
    )
  ) %>%
  select(-weather_low)



# ================================================================
# ----- 6) Cleaning Surface Variable (Binary Grass Variable) -----
# ================================================================

# ----- 6a) Normalize surface strings -----
# -----------------------------------------
# Keep original `surface` as `surface_raw` for reference
# Lowercase and trim whitespace → `surface_clean`
# Convert "" (empty strings) to NA
# Map known values into 2 labels: "grass" or "turf"

pbp_all <- pbp_all %>%
  mutate(
    # Preserve original surface column for auditing
    surface_raw = surface,
    
    # Lowercase + trim whitespace
    surface_clean = trimws(tolower(surface_raw)),
    
    # Convert "" → NA
    surface_clean = na_if(surface_clean, ""),
    
    # Standardize to two buckets where possible
    surface_clean = case_when(
      # Natural or hybrid grass
      surface_clean %in% c("grass", "dessograss") ~ "grass",
      
      # Common artificial turf variants
      surface_clean %in% c(
        "a_turf", "astroplay", "astroturf",
        "fieldturf", "matrixturf", "sportturf", "turf"
      ) ~ "turf",
      
      # Otherwise leave unchanged (maybe NA or an odd label)
      TRUE ~ surface_clean
    )
  )

# check for NA values
colSums(is.na(pbp_all[, c("surface_clean")]))


# ----- 6b) Handle Missing Values -----
# -------------------------------------

# For rows where `surface_clean` is NA, replace NA with the mode for that stadium
# and season combo. 

surface_mode_by_stadium <- pbp_all %>%
  # Only use rows where we know the cleaned surface
  filter(!is.na(surface_clean)) %>%
  group_by(stadium_abbr, season, surface_clean) %>%
  summarize(n = n(), .groups = "drop") %>%
  group_by(stadium_abbr, season) %>%
  slice_max(n, n = 1, with_ties = FALSE) %>%   # Take most common surface
  ungroup() %>%
  select(stadium_abbr, season, surface_mode = surface_clean)

pbp_all <- pbp_all %>%
  left_join(surface_mode_by_stadium,
            by = c("stadium_abbr", "season")) %>%
  mutate(
    # If original surface_clean is missing, fill with mode
    surface_final = if_else(
      is.na(surface_clean),
      surface_mode,
      surface_clean
    )
  ) %>%
  select(-surface_mode)

# check for NA values
colSums(is.na(pbp_all[, c("surface_final")]))
# all missing values accounted for


# ----- 6c) Create binary grass variable -----
# --------------------------------------------

pbp_all <- pbp_all %>%
  mutate(
    grass = case_when(
      surface_final == "grass" ~ 1,
      surface_final == "turf"  ~ 0,
      TRUE ~ NA_real_           # Should be extremely rare after imputation
    )
  )



# ======================================
# ----- 7) Ccreate Field Goal Data -----
# ======================================

# Extracting real field goal attempts and selecting all potential predictors

fg_data <- pbp_all %>%
  filter(field_goal_result %in% c("made", "missed", "blocked")) %>%
  mutate(
    fg_made = if_else(field_goal_result == "made", 1, 0),
    home    = if_else(posteam == home_team, 1, 0)
  ) %>%
  select(
    fg_made,
    kick_distance,
    down,
    ydstogo,
    qtr,
    game_seconds_remaining,
    score_differential,
    home,
    indoor,
    temp,
    wind,
    precipitation,
    grass
  )



# =====================================
# ----- 8) Check for Missing Data -----
# =====================================

print(colSums(is.na(fg_data)))

# Down has 5 missing values, but everything else is free of missing values
# since Field goals overwhelmingly occur on 4th down these 5 missing values can 
# be imputed with a 4 (which is the mode of the down variable)

fg_data <- fg_data %>%
  mutate(
    down = ifelse(is.na(down), 4, down)
  )


print(colSums(is.na(fg_data)))
# there is now no missing data



# =============================
# ----- 9) Exploring Data -----
# =============================

summary(fg_data)
view(fg_data)



# ===========================
# ----- 10) Save as CSV -----
# ===========================

output_path <- "nfl_fg_data_2001_2024.csv"

# Write the dataset to disk
write_csv(fg_data, output_path)

# Confirmation message
cat("Saved cleaned modeling dataset to:", output_path, "\n")