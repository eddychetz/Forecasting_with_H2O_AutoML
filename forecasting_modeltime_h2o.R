# BUSINESS SCIENCE LEARNING LABS ----
# MODELTIME H2O WORKSHOP ----
# **** ----

# BUSINESS OBJECTIVE ----
# - Forecast intermittent demand
# - Predict next 52-WEEKS
# **** ----

# Run This:
# remotes::install_github("business-science/modeltime.h2o")

# LIBRARIES ----

library(tidymodels) # ML such as `Scikit-learn` in Python
library(modeltime.h2o) # Auto ML
library(tidyverse) # Core
library(timetk) # TS analysis
library(lubridate)

# DATA ----

# Actual Sales demand data from `timetk` package

walmart_sales_weekly 

weekly_sales_tbl <- walmart_sales_weekly %>%
    select(id, Date, Weekly_Sales) %>%
    mutate(date = Date,
           sales = Weekly_Sales) %>%
    select(-c(Weekly_Sales, Date))

# * Time Plot ----

weekly_sales_tbl %>%
    group_by(id) %>%
    
    # From `timetk` package
    plot_time_series(
        .date_var = date,
        .value = sales,
        .facet_ncol = 3,
        .smooth = T,
        .smooth_period = "2 quarters",
        .interactive = T
    )

# * Seasonality Plot ----

ids <- unique(weekly_sales_tbl$id)

weekly_sales_tbl %>%
    filter(id == ids[3]) %>%
    
    plot_seasonal_diagnostics(
        .date_var = date,
        .value = log(sales)
    )

# TRAIN/TEST SPLITS ---

FORECAST_HORIZON <- 52 # no. of weeks

splits <-  time_series_split(
    weekly_sales_tbl,
    assess = FORECAST_HORIZON,
    cumulative = T
)

splits %>%
    tk_time_series_cv_plan() %>%
    plot_time_series_cv_plan(date, sales)

# PREPROCESSING ----

recipe_spec <- recipe(sales ~ ., data = training(splits)) %>%
    
    # Add engineered time calendar features
    step_timeseries_signature(date) %>%
    step_normalize(date_index.num, starts_with("date_year"))

#
recipe_spec %>% prep() %>% juice() %>% glimpse()

# MODELING ----

# Initial H2O
h2o.init(
    nthreads = -1,
    ip = 'localhost',
    port = 54321
)

# Optional (Turn off progress)
# h2o.no_progress()

# * Model Specification ----

model_spec_h2o <- automl_reg(mode  = 'regression') %>% # AutoML specification
    set_engine(
        engine                     = 'h2o',
        max_runtime_secs           = 30,
        max_runtime_secs_per_model = 10,
        max_models                 = 30,
        nfolds                     = 5,
        exclude_algos              =c("DeepLearning"), # Exclude DL models
        verbosity                  = NULL,
        seed                       = 786
    )

model_spec_h2o

# * Fitting ----
#  - This step will take some time depending on your model Specification selections

wflw_fit_h2o <- workflow() %>% # create workflow from `tidymodels` ecosystem
    add_model(model_spec_h2o) %>%
    add_recipe(recipe_spec) %>%
    fit(training(splits))

wflw_fit_h2o

# H2O AUTOML OBJECTS ----

# * H2O AutoML Leaderboard ----

wflw_fit_h2o %>% automl_leaderboard()

# * Saving /Loading Models ----

wflw_fit_h2o %>%
    
    # Update by picking another model from the leader board
    automl_update_model('GBM_grid_1_AutoML_16_20230626_160412_model_18') %>%
    save_h2o_model(path = 'h2o_models/GBM_grid_1_AutoML_16_20230626_160412_model_18')


load_h2o_model('h2o_models/GBM_grid_1_AutoML_16_20230626_160412_model_18/')

# FORECASTING ----

# * Modeltime Table ----

modeltime_tbl <- modeltime_table(
    wflw_fit_h2o,
    wflw_fit_h2o %>%
        automl_update_model('GBM_grid_1_AutoML_16_20230626_160412_model_18')
)

modeltime_tbl

# * Calibrate ----
# - Is actually a Residual Analysis

calibration_tbl <- modeltime_tbl %>%
    modeltime_calibrate(testing(splits))

calibration_tbl %>% 
    modeltime_accuracy() %>% 
    table_modeltime_accuracy()

# * Forecasting ----

calibration_tbl %>%
    modeltime_forecast(
        new_data = testing(splits),
        actual_data = weekly_sales_tbl,
        keep_data = T
    ) %>%
    group_by(id) %>%
    plot_modeltime_forecast(
        .facet_ncol = 3,
        .interactive = T
    )

# * Refitting ----

refit_tbl <- calibration_tbl%>%
    modeltime_refit(weekly_sales_tbl)

# * Future Forecast ----

future_tbl <- testing(splits) %>%
    group_by(id) %>%
    future_frame(date, .length_out = 52) %>%
    ungroup()

refit_tbl %>%
    modeltime_forecast(
        new_data = future_tbl,
        actual_data = weekly_sales_tbl,
        keep_data = T
    ) %>%
    group_by(id) %>%
    plot_modeltime_forecast(
        .facet_ncol = 3,
        .interactive = T
    )






