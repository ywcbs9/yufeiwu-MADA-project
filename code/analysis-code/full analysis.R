# Load packages
library(tidymodels)
library(dplyr)
library(lubridate)
library(embed)
library(here)
library(ggplot2)
library(stringr)
library(patchwork)
#install.packages("xgboost")
#install.packages("kernlab")


# Load data
LCAwage_Jan2024 <- readRDS(here("data", "processed-data", "LCAwage_Jan2024.rds"))

# Split the data into training (80%) and testing (20%) sets
set.seed(123)
data_split <- initial_split(LCAwage_Jan2024, prop = 0.8, strata = WAGE_CAT)
train_data <- training(data_split)
test_data  <- testing(data_split)


# 1. Random forest model
# --- Define the Recipe for the Random Forest Model with Default Hyperparameters ---
rf_default_class_recipe <- recipe(WAGE_CAT ~ SOC_TITLE + EMPLOYER_STATE + WORKSITE_STATE + EMPLOYER_NAME, data = train_data) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# --- Specify the Random Forest Model with Default Hyperparameters ---
# Defaults: mtry = floor(sqrt(p)), min_n = 1, trees = 500
rf_default_class_spec <- rand_forest() %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

# --- Create the Workflow ---
rf_default_class_workflow <- workflow() %>%
  add_recipe(rf_default_class_recipe) %>%
  add_model(rf_default_class_spec)

# --- Set Up Cross-Validation ---
# 5-fold cross-validation repeated 5 times (total of 25 resamples)
set.seed(123)
rf_default_cv_folds <- vfold_cv(train_data, v = 5, repeats = 5)

# --- Fit the Model with Cross-Validation ---
rf_default_cv_results <- fit_resamples(
  rf_default_class_workflow,
  resamples = rf_default_cv_folds,
  metrics = metric_set(accuracy)
)

# Collect and print cross-validation metrics
rf_default_cv_metrics <- collect_metrics(rf_default_cv_results)
print(rf_default_cv_metrics)

# --- Fit the Final Model on the Entire Training Data ---
rf_default_final_fit <- rf_default_class_workflow %>% fit(data = train_data)

# --- Evaluate on the Training Data ---
rf_default_train_predictions <- train_data %>% 
  bind_cols(
    predict(rf_default_final_fit, train_data),
    predict(rf_default_final_fit, train_data, type = "prob")
  )
rf_default_train_accuracy <- rf_default_train_predictions %>%
  accuracy(truth = WAGE_CAT, estimate = .pred_class) %>%
  mutate(dataset = "Train")

# --- Evaluate on the Test Data ---
rf_default_test_predictions <- test_data %>% 
  bind_cols(
    predict(rf_default_final_fit, test_data),
    predict(rf_default_final_fit, test_data, type = "prob")
  )
rf_default_test_accuracy <- rf_default_test_predictions %>%
  accuracy(truth = WAGE_CAT, estimate = .pred_class) %>%
  mutate(dataset = "Test")

# --- Combine the Accuracy Metrics ---
rf_default_combined_accuracy <- bind_rows(rf_default_train_accuracy, rf_default_test_accuracy) %>%
  select(dataset, .metric, .estimator, .estimate)
print(rf_default_combined_accuracy)

# Save the combined accuracy results as an RDS file
saveRDS(rf_default_combined_accuracy, file = "results/tables/rf_combined_accuracy_cv_default.rds")



# Extract per‑resample accuracies

cv_acc_resample <- rf_default_cv_results %>%
  collect_metrics(summarize = FALSE) %>% 
  filter(.metric == "accuracy") %>% 
  mutate(
    fold = str_extract(id, "(?<=Fold)\\d+"),     # Fold 1 … 5
    rep  = str_extract(id, "(?<=Repeat)\\d+")    # Repeat 1 … 5
  )



# Try different models to see if the model performance can be improved

# 2. XGBoot model
# --- Define the Recipe for the XGBoost Model with Regularization ---
# Use all predictors and one-hot encode them
xgb_reg_class_recipe <- recipe(WAGE_CAT ~ SOC_TITLE + EMPLOYER_STATE + WORKSITE_STATE + EMPLOYER_NAME, data = train_data) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# --- Specify the XGBoost Model with Regularization ---
# Regularization parameters (lambda and alpha) are now passed via set_engine()
xgb_reg_class_spec <- boost_tree(
  trees         = 500,
  tree_depth    = 6,
  learn_rate    = 0.3,
  loss_reduction = 1   # corresponds to gamma in xgboost
) %>% 
  set_engine("xgboost", lambda = 1, alpha = 0.5) %>% 
  set_mode("classification")

# --- Create the Workflow ---
xgb_reg_class_workflow <- workflow() %>%
  add_recipe(xgb_reg_class_recipe) %>%
  add_model(xgb_reg_class_spec)

# --- Set Up Cross-Validation ---
# 5-fold cross-validation repeated 5 times (total of 25 resamples)
set.seed(123)
xgb_reg_cv_folds <- vfold_cv(train_data, v = 5, repeats = 5)

# --- Fit the Model with Cross-Validation ---
xgb_reg_cv_results <- fit_resamples(
  xgb_reg_class_workflow,
  resamples = xgb_reg_cv_folds,
  metrics = metric_set(accuracy)
)

# Collect and print cross-validation metrics
xgb_reg_cv_metrics <- collect_metrics(xgb_reg_cv_results)
print(xgb_reg_cv_metrics)

# --- Fit the Final Model on the Entire Training Data ---
xgb_reg_final_fit <- xgb_reg_class_workflow %>% fit(data = train_data)

# --- Evaluate on the Training Data ---
xgb_reg_train_predictions <- train_data %>% 
  bind_cols(
    predict(xgb_reg_final_fit, train_data),
    predict(xgb_reg_final_fit, train_data, type = "prob")
  )
xgb_reg_train_accuracy <- xgb_reg_train_predictions %>%
  accuracy(truth = WAGE_CAT, estimate = .pred_class) %>%
  mutate(dataset = "Train")

# --- Evaluate on the Test Data ---
xgb_reg_test_predictions <- test_data %>% 
  bind_cols(
    predict(xgb_reg_final_fit, test_data),
    predict(xgb_reg_final_fit, test_data, type = "prob")
  )
xgb_reg_test_accuracy <- xgb_reg_test_predictions %>%
  accuracy(truth = WAGE_CAT, estimate = .pred_class) %>%
  mutate(dataset = "Test")

# --- Combine the Accuracy Metrics ---
xgb_reg_combined_accuracy <- bind_rows(xgb_reg_train_accuracy, xgb_reg_test_accuracy) %>%
  select(dataset, .metric, .estimator, .estimate)
print(xgb_reg_combined_accuracy)

# save as rds
saveRDS(xgb_reg_combined_accuracy, file = "results/tables/combined_accuracy_cv_xgb_reg.rds")


# Per‑resample accuracies (25 rows)
xgb_cv_acc_resample <- xgb_reg_cv_results %>%
  collect_metrics(summarize = FALSE) %>%          # keep every resample
  filter(.metric == "accuracy") %>% 
  mutate(
    fold = str_extract(id, "(?<=Fold)\\d+"),      # Fold 1 … 5
    rep  = str_extract(id, "(?<=Repeat)\\d+")     # Repeat 1 … 5
  )





# 3. SVM
# --- Define the Recipe for the SVM Model ---
# Use all predictors, one-hot encode them, and normalize the numeric predictors.
svm_class_recipe <- recipe(WAGE_CAT ~ SOC_TITLE + EMPLOYER_STATE + WORKSITE_STATE + EMPLOYER_NAME, data = train_data) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_normalize(all_predictors())

# --- Specify the SVM Model ---
# Using an RBF kernel via the kernlab engine with default parameters.
svm_class_spec <- svm_rbf() %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

# --- Create the Workflow ---
svm_class_workflow <- workflow() %>%
  add_recipe(svm_class_recipe) %>%
  add_model(svm_class_spec)

# --- Set Up Cross-Validation ---
# 5-fold cross-validation repeated 5 times (total of 25 resamples)
set.seed(123)
svm_cv_folds <- vfold_cv(train_data, v = 5, repeats = 5)

# --- Fit the Model with Cross-Validation ---
svm_cv_results <- fit_resamples(
  svm_class_workflow,
  resamples = svm_cv_folds,
  metrics = metric_set(accuracy)
)

# Collect and print cross-validation metrics
svm_cv_metrics <- collect_metrics(svm_cv_results)
print(svm_cv_metrics)

# --- Fit the Final Model on the Entire Training Data ---
svm_final_fit <- svm_class_workflow %>% fit(data = train_data)

# --- Evaluate on the Training Data ---
svm_train_predictions <- train_data %>% 
  bind_cols(
    predict(svm_final_fit, train_data),
    predict(svm_final_fit, train_data, type = "prob")
  )
svm_train_accuracy <- svm_train_predictions %>%
  accuracy(truth = WAGE_CAT, estimate = .pred_class) %>%
  mutate(dataset = "Train")

# --- Evaluate on the Test Data ---
svm_test_predictions <- test_data %>% 
  bind_cols(
    predict(svm_final_fit, test_data),
    predict(svm_final_fit, test_data, type = "prob")
  )
svm_test_accuracy <- svm_test_predictions %>%
  accuracy(truth = WAGE_CAT, estimate = .pred_class) %>%
  mutate(dataset = "Test")

# --- Combine the Accuracy Metrics ---
svm_combined_accuracy <- bind_rows(svm_train_accuracy, svm_test_accuracy) %>%
  select(dataset, .metric, .estimator, .estimate)
print(svm_combined_accuracy)

# save as rds
saveRDS(svm_combined_accuracy, file = "results/tables/combined_accuracy_cv_svm.rds")


# Make plot
# Add a column to indicate the model type for each set of results
rf_acc   <- rf_default_combined_accuracy %>% mutate(model = "Random Forest")
xgb_acc  <- xgb_reg_combined_accuracy %>% mutate(model = "XGBoost")
svm_acc  <- svm_combined_accuracy %>% mutate(model = "SVM")

# Combine the three data frames
combined_model_accuracy <- bind_rows(rf_acc, xgb_acc, svm_acc)

# Create a bar plot comparing performance by dataset and model
model_compare <- ggplot(combined_model_accuracy, aes(x = dataset, y = .estimate, fill = model)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  labs(
    title = "Comparison of Model Performance: Random Forest, XGBoost, and SVM",
    x = "Dataset",
    y = "Accuracy"
  ) +
  theme_minimal()

# Save the plot to a file
ggsave("results/figures/model_compare.png", plot = model_compare, width = 8, height = 6)


# Per‑resample accuracies (25 rows)
svm_cv_acc_resample <- svm_cv_results %>%
  collect_metrics(summarize = FALSE) %>%          # keep every resample
  filter(.metric == "accuracy") %>% 
  mutate(
    fold = str_extract(id, "(?<=Fold)\\d+"),      # Fold 1 … 5
    rep  = str_extract(id, "(?<=Repeat)\\d+")     # Repeat 1 … 5
  )



# Helper: a simple box + jitter plot of .estimate
make_box_plot <- function(df, model_label) {
  ggplot(df, aes(x = "", y = .estimate)) +
    geom_boxplot(width = 0.25, outlier.shape = NA) +
    geom_jitter(width = 0.05, height = 0, alpha = 0.7) +
    labs(
      title = sprintf("Distribution of %s CV accuracies", model_label),
      x = NULL, 
      y = "Accuracy"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank()
    )
}

# 1. Random Forest accuracy distribution
#    (assuming you have rf_default_cv_results from fit_resamples(...))
rf_cv_acc_resample <- rf_default_cv_results %>%
  collect_metrics(summarize = FALSE) %>%
  filter(.metric == "accuracy")

p_rf <- make_box_plot(rf_cv_acc_resample, "Random Forest")

# 2. XGBoost accuracy distribution
xgb_cv_acc_resample <- xgb_reg_cv_results %>% 
  collect_metrics(summarize = FALSE) %>%
  filter(.metric == "accuracy")

p_xgb <- make_box_plot(xgb_cv_acc_resample, "XGBoost")

# 3. SVM accuracy distribution
svm_cv_acc_resample <- svm_cv_results %>%
  collect_metrics(summarize = FALSE) %>%
  filter(.metric == "accuracy")

p_svm <- make_box_plot(svm_cv_acc_resample, "SVM")

# Combine in a single row (3 columns)
fold_panel <- p_rf | p_xgb | p_svm

# Show in viewer / notebook
fold_panel

# Save to file
ggsave(
  filename = "results/figures/cv_accuracy_distribution_panel.png",
  plot     = fold_panel,
  width    = 12,  # in inches
  height   = 4,
  dpi      = 300
)

