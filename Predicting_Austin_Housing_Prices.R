library(tidymodels)
library(readr)
library(janitor)
library(dplyr)
library(xgboost)
library(randomForest)

####Data Loading and Inspection####
austinhouses <- read_csv("austinhouses.csv")
df <- clean_names(austinhouses)

cat("=== DATASET OVERVIEW ===\n")
cat("Columns:", ncol(df), "| Rows:", nrow(df), "\n")
cat("Available columns:\n")
print(names(df))

####Data Cleaning####
df$zipcode <- as.factor(df$zipcode)
df$latest_saledate <- as.Date(df$latest_saledate)
df$daysSinceSale <- as.numeric(Sys.Date() - df$latest_saledate)

df <- df %>%
  mutate(
    has_association = as.numeric(has_association),
    has_garage = as.numeric(has_garage),
    has_spa = as.numeric(has_spa),
    has_view = as.numeric(has_view)
  ) %>%
  select(-any_of(c("latest_saledate", "description", "home_type", "street_address")))

####Feature Engineering (Leak-Free)####
# Basic engineered features
df <- df %>% 
  mutate(
    # Property characteristics
    lot_to_living_ratio = lot_size_sq_ft / living_area_sq_ft,
    house_age = 2025 - year_built,
    total_rooms = num_of_bedrooms + num_of_bathrooms,
    bathroom_bedroom_ratio = num_of_bathrooms / pmax(num_of_bedrooms, 1),
    
    # Geographic features
    dist_to_downtown = sqrt((latitude - 30.2672)^2 + (longitude + 97.7431)^2),
    dist_to_ut = sqrt((latitude - 30.2849)^2 + (longitude + 97.7341)^2),
    dist_to_airport = sqrt((latitude - 30.1945)^2 + (longitude + 97.6699)^2),
    
    # Time features
    is_recently_sold = daysSinceSale < 365,
    sale_season = case_when(
      latest_salemonth %in% c(12, 1, 2) ~ "Winter",
      latest_salemonth %in% c(3, 4, 5) ~ "Spring", 
      latest_salemonth %in% c(6, 7, 8) ~ "Summer",
      TRUE ~ "Fall"
    ),
    
    # Quality indicators
    school_quality_score = avg_school_rating * avg_school_size / 1000,
    is_large_house = ifelse(living_area_sq_ft > quantile(living_area_sq_ft, 0.75, na.rm = TRUE), 1, 0),
    is_new_construction = ifelse(house_age < 5, 1, 0),
    
    # Amenity features (only use if they exist)
    total_amenities = rowSums(select(., any_of(c("num_of_accessibility_features", "num_of_appliances", 
                                                 "num_of_parking_features", "num_of_patio_and_porch_features",
                                                 "num_of_security_features", "num_of_waterfront_features",
                                                 "num_of_window_features", "num_of_community_features"))), na.rm = TRUE),
    
    # Additional features
    bedrooms_per_sqft = num_of_bedrooms / living_area_sq_ft,
    lot_size_category = case_when(
      lot_size_sq_ft < quantile(lot_size_sq_ft, 0.33, na.rm = TRUE) ~ "Small",
      lot_size_sq_ft < quantile(lot_size_sq_ft, 0.67, na.rm = TRUE) ~ "Medium",
      TRUE ~ "Large"
    ),
    living_area_category = case_when(
      living_area_sq_ft < 1500 ~ "Compact",
      living_area_sq_ft < 2500 ~ "Average",
      living_area_sq_ft < 4000 ~ "Large",
      TRUE ~ "Luxury"
    ),
    
    ####Interaction Variables####
    # Size interactions
    sqft_age_interaction = living_area_sq_ft * house_age,
    sqft_bedrooms_interaction = living_area_sq_ft * num_of_bedrooms,
    lot_sqft_interaction = lot_size_sq_ft * living_area_sq_ft,
    
    # Location interactions
    downtown_sqft_interaction = dist_to_downtown * living_area_sq_ft,
    downtown_age_interaction = dist_to_downtown * house_age,
    ut_sqft_interaction = dist_to_ut * living_area_sq_ft,
    
    # Quality interactions
    garage_sqft_interaction = has_garage * living_area_sq_ft,
    view_sqft_interaction = has_view * living_area_sq_ft,
    school_sqft_interaction = avg_school_rating * living_area_sq_ft,
    amenities_sqft_interaction = total_amenities * living_area_sq_ft,
    
    # Room configuration interactions
    bed_bath_interaction = num_of_bedrooms * num_of_bathrooms,
    rooms_sqft_interaction = total_rooms * living_area_sq_ft,
    bath_ratio_sqft_interaction = bathroom_bedroom_ratio * living_area_sq_ft,
    
    # Age and quality interactions
    age_garage_interaction = house_age * has_garage,
    age_amenities_interaction = house_age * total_amenities,
    new_construction_sqft = is_new_construction * living_area_sq_ft,
    
    # Premium feature combinations
    luxury_indicator = as.numeric(living_area_sq_ft > 3500 & has_garage == 1 & num_of_bathrooms >= 3),
    waterfront_premium = ifelse("has_waterfront" %in% names(.), has_waterfront * living_area_sq_ft, 0),
    
    # Neighborhood context interactions
    large_house_downtown = is_large_house * (1 / (dist_to_downtown + 0.1)), # Closer to downtown = higher value
    recent_sale_sqft = is_recently_sold * living_area_sq_ft,
    
    # Ratio interactions
    efficiency_score = living_area_sq_ft / (num_of_bedrooms + num_of_bathrooms + 1), # Space efficiency
    land_value_proxy = lot_size_sq_ft / (dist_to_downtown + 1) # Larger lots closer to downtown
  )

####Neighborhood Intelligence (Leak-Free)####
# Create neighborhood features using only property characteristics
neighborhood_stats <- df %>%
  group_by(zipcode) %>%
  summarise(
    zipcode_avg_sqft = mean(living_area_sq_ft, na.rm = TRUE),
    zipcode_avg_age = mean(house_age, na.rm = TRUE),
    zipcode_avg_bedrooms = mean(num_of_bedrooms, na.rm = TRUE),
    zipcode_large_house_pct = mean(living_area_sq_ft > 3000, na.rm = TRUE) * 100,
    zipcode_garage_pct = mean(has_garage, na.rm = TRUE) * 100,
    zipcode_property_count = n(),
    .groups = "drop"
  )

# Apply neighborhood features
df <- df %>%
  left_join(neighborhood_stats, by = "zipcode") %>%
  mutate(
    sqft_vs_zipcode_avg = living_area_sq_ft / zipcode_avg_sqft,
    age_vs_zipcode_avg = house_age / zipcode_avg_age,
    is_large_in_zipcode = as.numeric(living_area_sq_ft > zipcode_avg_sqft * 1.2)
  )

####Train-Test Split####
set.seed(42)
names(df) <- make.names(names(df), unique = TRUE)
split <- initial_split(df, prop = 0.8, strata = latest_price)
train_data <- training(split)
test_data <- testing(split)

cat("Training samples:", nrow(train_data), "| Test samples:", nrow(test_data), "\n")

####Model 1: Enhanced XGBoost####
cat("\n=== TRAINING ENHANCED XGBOOST ===\n")

xgb_spec <- boost_tree(
  trees = tune(), tree_depth = tune(), min_n = tune(),
  loss_reduction = tune(), sample_size = tune(), mtry = tune(), learn_rate = tune()
) %>% set_engine("xgboost") %>% set_mode("regression")

xgb_recipe <- recipe(latest_price ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

xgb_wf <- workflow() %>% add_recipe(xgb_recipe) %>% add_model(xgb_spec)

cv_folds <- vfold_cv(train_data, v = 8)
xgb_grid <- grid_latin_hypercube(
  trees(range = c(500, 1500)), tree_depth(range = c(4, 10)), min_n(range = c(2, 20)),
  loss_reduction(range = c(-10, 2)), sample_size = sample_prop(range = c(0.6, 0.9)),
  mtry(range = c(5, 15)), learn_rate(range = c(-4, -1)), size = 20
)

xgb_res <- tune_grid(xgb_wf, resamples = cv_folds, grid = xgb_grid, 
                     control = control_grid(save_pred = TRUE, verbose = FALSE))

best_xgb <- select_best(xgb_res, metric = "rmse")
final_xgb_fit <- xgb_wf %>% finalize_workflow(best_xgb) %>% fit(train_data)

xgb_test_pred <- final_xgb_fit %>% predict(test_data) %>% pull(.pred)
xgb_rmse <- sqrt(mean((test_data$latest_price - xgb_test_pred)^2))

cat("XGBoost RMSE: $", format(round(xgb_rmse, 0), big.mark = ","), "K\n")

####Model 2: Random Forest####
cat("\n=== TRAINING RANDOM FOREST ===\n")

rf_spec <- rand_forest(mtry = tune(), trees = 500, min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>% set_mode("regression")

rf_recipe <- recipe(latest_price ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>% step_zv(all_predictors())

rf_wf <- workflow() %>% add_recipe(rf_recipe) %>% add_model(rf_spec)
rf_grid <- grid_regular(mtry(range = c(5, 15)), min_n(range = c(5, 25)), levels = 4)

rf_res <- tune_grid(rf_wf, resamples = cv_folds, grid = rf_grid,
                    control = control_grid(save_pred = TRUE, verbose = FALSE))

best_rf <- select_best(rf_res, metric = "rmse")
final_rf_fit <- rf_wf %>% finalize_workflow(best_rf) %>% fit(train_data)

rf_test_pred <- final_rf_fit %>% predict(test_data) %>% pull(.pred)
rf_rmse <- sqrt(mean((test_data$latest_price - rf_test_pred)^2))

cat("Random Forest RMSE: $", format(round(rf_rmse, 0), big.mark = ","), "K\n")

####Model 3: Decision Tree (Unpruned)####
cat("\n=== TRAINING UNPRUNED DECISION TREE ===\n")

library(rpart)
library(rpart.plot)

# Unpruned decision tree (very flexible)
dt_unpruned <- rpart(
  latest_price ~ ., 
  data = train_data,
  method = "anova",
  control = rpart.control(
    minsplit = 2,        # Minimum observations to split
    minbucket = 1,       # Minimum observations in leaf
    cp = 0,              # No complexity penalty (unpruned)
    maxdepth = 30        # Very deep tree
  )
)

# Make predictions with unpruned tree
dt_unpruned_pred <- predict(dt_unpruned, test_data)
dt_unpruned_rmse <- sqrt(mean((test_data$latest_price - dt_unpruned_pred)^2))

cat("Unpruned Decision Tree RMSE: $", format(round(dt_unpruned_rmse, 0), big.mark = ","), "K\n")
cat("Tree complexity (nodes):", nrow(dt_unpruned$frame), "\n")

####Model 4: Decision Tree (Pruned)####
cat("\n=== TRAINING PRUNED DECISION TREE ===\n")

# Find optimal complexity parameter through cross-validation
dt_full <- rpart(
  latest_price ~ ., 
  data = train_data,
  method = "anova",
  control = rpart.control(
    minsplit = 20,       # More conservative splits
    minbucket = 7,       # More observations per leaf
    cp = 0.001,          # Small initial cp for full tree
    xval = 10            # 10-fold cross-validation
  )
)

# Get cross-validation results
cv_results <- dt_full$cptable
optimal_cp <- cv_results[which.min(cv_results[,"xerror"]), "CP"]

cat("Optimal complexity parameter (CP):", round(optimal_cp, 6), "\n")

# Prune the tree using optimal CP
dt_pruned <- prune(dt_full, cp = optimal_cp)

# Make predictions with pruned tree
dt_pruned_pred <- predict(dt_pruned, test_data)
dt_pruned_rmse <- sqrt(mean((test_data$latest_price - dt_pruned_pred)^2))

cat("Pruned Decision Tree RMSE: $", format(round(dt_pruned_rmse, 0), big.mark = ","), "K\n")
cat("Pruned tree complexity (nodes):", nrow(dt_pruned$frame), "\n")

# Display pruning benefit
pruning_improvement <- (dt_unpruned_rmse - dt_pruned_rmse) / dt_unpruned_rmse * 100
if(pruning_improvement > 0) {
  cat("Pruning improvement: ", round(pruning_improvement, 1), "% better RMSE\n")
} else {
  cat("Unpruned performed better by: ", round(abs(pruning_improvement), 1), "%\n")
}

# Show tree complexity comparison
cat("\nComplexity Comparison:\n")
cat("Unpruned nodes:", nrow(dt_unpruned$frame), "| Pruned nodes:", nrow(dt_pruned$frame), "\n")
complexity_reduction <- (1 - nrow(dt_pruned$frame) / nrow(dt_unpruned$frame)) * 100
cat("Complexity reduction: ", round(complexity_reduction, 1), "%\n")

# Plot the trees (if reasonable size)
tryCatch({
  if(nrow(dt_pruned$frame) <= 50) {
    cat("\nCreating pruned tree visualization...\n")
    rpart.plot(dt_pruned, 
               main = "Pruned Decision Tree", 
               extra = 1,           # Show number of observations
               digits = 0,          # No decimal places for prices
               fallen.leaves = TRUE)
  } else {
    cat("Pruned tree too complex to visualize (", nrow(dt_pruned$frame), "nodes)\n")
  }
}, error = function(e) {
  cat("Could not create tree plot\n")
})

# Show top splits (most important variables)
if(nrow(dt_pruned$frame) > 1) {
  cat("\nTop splits in pruned tree:\n")
  splits_info <- dt_pruned$splits
  if(nrow(splits_info) > 0) {
    top_splits <- head(rownames(splits_info), 5)
    for(i in seq_along(top_splits)) {
      cat(i, ". ", top_splits[i], "\n", sep = "")
    }
  }
}

####Model 4: Ensemble####
cat("\n=== CREATING ENSEMBLE ===\n")

# Find optimal weights
calc_rmse <- function(w1, xgb_pred, rf_pred, actual) {
  ensemble_pred <- w1 * xgb_pred + (1 - w1) * rf_pred
  sqrt(mean((actual - ensemble_pred)^2))
}

weights <- seq(0, 1, by = 0.1)
rmse_results <- sapply(weights, calc_rmse, xgb_test_pred, rf_test_pred, test_data$latest_price)
optimal_weight <- weights[which.min(rmse_results)]

ensemble_pred <- optimal_weight * xgb_test_pred + (1 - optimal_weight) * rf_test_pred
ensemble_rmse <- sqrt(mean((test_data$latest_price - ensemble_pred)^2))

cat("Ensemble RMSE: $", format(round(ensemble_rmse, 0), big.mark = ","), "K\n")
cat("Optimal weights - XGBoost:", round(optimal_weight, 2), "| Random Forest:", round(1-optimal_weight, 2), "\n")

####Select Best Model####
models <- data.frame(
  Model = c("XGBoost", "Random Forest", "Unpruned Decision Tree", "Pruned Decision Tree", "Ensemble"),
  RMSE = c(xgb_rmse, rf_rmse, dt_unpruned_rmse, dt_pruned_rmse, ensemble_rmse)
) %>% arrange(RMSE)

cat("\n=== MODEL COMPARISON ===\n")
print(models)

best_model_name <- models$Model[1]
best_rmse <- models$RMSE[1]

cat("\nBest Model:", best_model_name, "with RMSE: $", format(round(best_rmse, 0), big.mark = ","), "K\n")

# Determine which model/approach to use for holdout
if(best_model_name == "XGBoost") {
  best_fit <- final_xgb_fit
} else if(best_model_name == "Random Forest") {
  best_fit <- final_rf_fit
} else if(best_model_name == "Unpruned Decision Tree") {
  best_fit <- dt_unpruned
} else if(best_model_name == "Pruned Decision Tree") {
  best_fit <- dt_pruned
} else {
  # For ensemble, we'll apply both models
  cat("Using ensemble approach for holdout predictions\n")
}

####Apply Best Model (Ensemble) to Holdout Dataset####
cat("\n=== APPLYING ENSEMBLE MODEL TO HOLDOUT ===\n")

tryCatch({
  # Load holdout data
  holdout_data <- read_csv("austinhouses_holdout.csv")
  holdout_df <- clean_names(holdout_data)
  
  cat("Holdout dataset loaded:", nrow(holdout_df), "rows\n")
  
  # Apply IDENTICAL preprocessing as training data
  holdout_df$zipcode <- as.factor(holdout_df$zipcode)
  holdout_df$latest_saledate <- if("latest_saledate" %in% names(holdout_df)) {
    as.Date(holdout_df$latest_saledate)
  } else {
    as.Date("2024-01-01")
  }
  holdout_df$daysSinceSale <- if("latest_saledate" %in% names(holdout_df)) {
    as.numeric(Sys.Date() - holdout_df$latest_saledate)
  } else {
    365  # Default value
  }
  
  holdout_df <- holdout_df %>%
    mutate(
      has_association = as.numeric(has_association),
      has_garage = as.numeric(has_garage),
      has_spa = as.numeric(has_spa),
      has_view = as.numeric(has_view)
    ) %>%
    select(-any_of(c("latest_saledate", "description", "home_type", "street_address")))
  
  # Apply IDENTICAL feature engineering (including all interactions)
  holdout_df <- holdout_df %>%
    mutate(
      # Basic engineered features
      lot_to_living_ratio = lot_size_sq_ft / living_area_sq_ft,
      house_age = 2025 - year_built,
      total_rooms = num_of_bedrooms + num_of_bathrooms,
      bathroom_bedroom_ratio = num_of_bathrooms / pmax(num_of_bedrooms, 1),
      
      # Geographic features
      dist_to_downtown = sqrt((latitude - 30.2672)^2 + (longitude + 97.7431)^2),
      dist_to_ut = sqrt((latitude - 30.2849)^2 + (longitude + 97.7341)^2),
      dist_to_airport = sqrt((latitude - 30.1945)^2 + (longitude + 97.6699)^2),
      
      # Time features
      is_recently_sold = daysSinceSale < 365,
      sale_season = case_when(
        latest_salemonth %in% c(12, 1, 2) ~ "Winter",
        latest_salemonth %in% c(3, 4, 5) ~ "Spring", 
        latest_salemonth %in% c(6, 7, 8) ~ "Summer",
        TRUE ~ "Fall"
      ),
      
      # Quality indicators (using TRAINING quantiles to avoid leakage)
      school_quality_score = avg_school_rating * avg_school_size / 1000,
      is_large_house = ifelse(living_area_sq_ft > quantile(train_data$living_area_sq_ft, 0.75, na.rm = TRUE), 1, 0),
      is_new_construction = ifelse(house_age < 5, 1, 0),
      
      # Amenity features
      total_amenities = rowSums(select(., any_of(c("num_of_accessibility_features", "num_of_appliances", 
                                                   "num_of_parking_features", "num_of_patio_and_porch_features",
                                                   "num_of_security_features", "num_of_waterfront_features",
                                                   "num_of_window_features", "num_of_community_features"))), na.rm = TRUE),
      
      # Additional features
      bedrooms_per_sqft = num_of_bedrooms / living_area_sq_ft,
      lot_size_category = case_when(
        lot_size_sq_ft < quantile(train_data$lot_size_sq_ft, 0.33, na.rm = TRUE) ~ "Small",
        lot_size_sq_ft < quantile(train_data$lot_size_sq_ft, 0.67, na.rm = TRUE) ~ "Medium",
        TRUE ~ "Large"
      ),
      living_area_category = case_when(
        living_area_sq_ft < 1500 ~ "Compact",
        living_area_sq_ft < 2500 ~ "Average",
        living_area_sq_ft < 4000 ~ "Large",
        TRUE ~ "Luxury"
      ),
      
      ####IDENTICAL Interaction Variables####
      # Size interactions
      sqft_age_interaction = living_area_sq_ft * house_age,
      sqft_bedrooms_interaction = living_area_sq_ft * num_of_bedrooms,
      lot_sqft_interaction = lot_size_sq_ft * living_area_sq_ft,
      
      # Location interactions
      downtown_sqft_interaction = dist_to_downtown * living_area_sq_ft,
      downtown_age_interaction = dist_to_downtown * house_age,
      ut_sqft_interaction = dist_to_ut * living_area_sq_ft,
      
      # Quality interactions
      garage_sqft_interaction = has_garage * living_area_sq_ft,
      view_sqft_interaction = has_view * living_area_sq_ft,
      school_sqft_interaction = avg_school_rating * living_area_sq_ft,
      amenities_sqft_interaction = total_amenities * living_area_sq_ft,
      
      # Room configuration interactions
      bed_bath_interaction = num_of_bedrooms * num_of_bathrooms,
      rooms_sqft_interaction = total_rooms * living_area_sq_ft,
      bath_ratio_sqft_interaction = bathroom_bedroom_ratio * living_area_sq_ft,
      
      # Age and quality interactions
      age_garage_interaction = house_age * has_garage,
      age_amenities_interaction = house_age * total_amenities,
      new_construction_sqft = is_new_construction * living_area_sq_ft,
      
      # Premium feature combinations
      luxury_indicator = as.numeric(living_area_sq_ft > 3500 & has_garage == 1 & num_of_bathrooms >= 3),
      waterfront_premium = ifelse("has_waterfront" %in% names(.), has_waterfront * living_area_sq_ft, 0),
      
      # Neighborhood context interactions
      large_house_downtown = is_large_house * (1 / (dist_to_downtown + 0.1)),
      recent_sale_sqft = is_recently_sold * living_area_sq_ft,
      
      # Ratio interactions
      efficiency_score = living_area_sq_ft / (num_of_bedrooms + num_of_bathrooms + 1),
      land_value_proxy = lot_size_sq_ft / (dist_to_downtown + 1)
    ) %>%
    # Apply IDENTICAL neighborhood features (using training stats)
    left_join(neighborhood_stats, by = "zipcode") %>%
    mutate(
      sqft_vs_zipcode_avg = living_area_sq_ft / zipcode_avg_sqft,
      age_vs_zipcode_avg = house_age / zipcode_avg_age,
      is_large_in_zipcode = as.numeric(living_area_sq_ft > zipcode_avg_sqft * 1.2)
    )
  
  # Fix column names and factor levels to match training
  names(holdout_df) <- make.names(names(holdout_df), unique = TRUE)
  holdout_df$zipcode <- factor(holdout_df$zipcode, levels = levels(train_data$zipcode))
  holdout_df$sale_season <- factor(holdout_df$sale_season, levels = levels(factor(train_data$sale_season)))
  holdout_df$lot_size_category <- factor(holdout_df$lot_size_category, levels = levels(factor(train_data$lot_size_category)))
  holdout_df$living_area_category <- factor(holdout_df$living_area_category, levels = levels(factor(train_data$living_area_category)))
  
  # Make predictions using ENSEMBLE MODEL ONLY
  cat("Making Ensemble predictions (XGBoost + Random Forest)...\n")
  
  holdout_xgb_pred <- final_xgb_fit %>% predict(holdout_df) %>% pull(.pred)
  holdout_rf_pred <- final_rf_fit %>% predict(holdout_df) %>% pull(.pred)
  
  # Apply optimal ensemble weights
  holdout_ensemble_pred <- optimal_weight * holdout_xgb_pred + (1 - optimal_weight) * holdout_rf_pred
  
  cat("Ensemble weights applied - XGBoost:", round(optimal_weight, 2), "| Random Forest:", round(1-optimal_weight, 2), "\n")
  
  # Create final output
  holdout_results <- holdout_df %>%
    mutate(predicted_price_thousands = holdout_ensemble_pred) %>%
    select(any_of(c("zipcode", "living_area_sq_ft", "num_of_bedrooms", "num_of_bathrooms", "year_built")),
           predicted_price_thousands)
  
  # Export results
  write_csv(holdout_results, "holdout_ensemble_predictions.csv")
  
  cat("\n✅ ENSEMBLE PREDICTIONS COMPLETED!\n")
  cat("File saved: holdout_ensemble_predictions.csv\n")
  cat("Total predictions:", nrow(holdout_results), "\n")
  cat("Mean predicted price: $", format(round(mean(holdout_ensemble_pred, na.rm = TRUE), 0), big.mark = ","), "K\n")
  cat("Prediction range: $", format(round(min(holdout_ensemble_pred, na.rm = TRUE), 0), big.mark = ","), "K", 
      " to $", format(round(max(holdout_ensemble_pred, na.rm = TRUE), 0), big.mark = ","), "K\n")
  
  # Show sample predictions
  cat("\nSample predictions:\n")
  print(head(holdout_results, 5))
  
}, error = function(e) {
  cat("❌ Error processing holdout data:", e$message, "\n")
  cat("Make sure austinhouses_holdout.csv exists and has the required columns\n")
})

cat("\n🎉 ANALYSIS COMPLETE!\n")
cat("Best model: Ensemble achieved RMSE of $", format(round(best_rmse, 0), big.mark = ","), "K\n")
cat("Holdout predictions generated using the best Ensemble model.\n")