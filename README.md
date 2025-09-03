# Predicting Austin Housing Prices

A machine learning project that predicts Austin, Texas housing prices using multiple regression models and ensemble methods.

## 📊 Project Overview

This project analyzes Austin housing market data to build predictive models for property values. The analysis includes comprehensive feature engineering, multiple model comparisons, and ensemble methods to achieve optimal prediction accuracy.

**Key Results:**
- **Best Model**: Ensemble (XGBoost + Random Forest)
- **Performance**: Achieved competitive RMSE on test data
- **Features**: 50+ engineered features including location proximity, property characteristics, and neighborhood intelligence

---

## 📋 Dataset Information

### Training Data (`austinhouses.csv`)
- **Properties**: Austin residential real estate listings
- **Features**: Property characteristics, location data, school information, amenities
- **Target Variable**: `latest_price` (in thousands of dollars)

### Key Features Include:
- **Property Details**: Square footage, bedrooms, bathrooms, year built, lot size
- **Location**: Latitude, longitude, zipcode
- **Amenities**: Garage, spa, view, association, feature counts
- **Schools**: Average rating, distance, size, student-teacher ratios
- **Market Data**: Sale dates, property tax rates

---

## 🔧 Feature Engineering

The analysis includes extensive feature engineering:

### Basic Features
- `house_age`: Calculated from year built  
- `total_rooms`: Bedrooms + bathrooms  
- `lot_to_living_ratio`: Lot size relative to living space  
- `bathroom_bedroom_ratio`: Bathroom to bedroom ratio  

### Location Intelligence
- `dist_to_downtown`: Distance to Austin downtown (30.2672, -97.7431)  
- `dist_to_ut`: Distance to UT campus (30.2849, -97.7341)  
- `dist_to_airport`: Distance to Austin airport (30.1945, -97.6699)  

### Neighborhood Features
- Zipcode-based statistics (average square footage, age, amenities)  
- Property performance relative to neighborhood averages  
- Local market indicators  

### Interaction Variables
- Size × Age interactions  
- Location × Property characteristics  
- Quality × Size interactions  
- Premium feature combinations  

---

## 🤖 Models Implemented

### 1. Enhanced XGBoost
- Hyperparameter tuning with Latin hypercube search  
- 8-fold stratified cross-validation  
- Dummy encoding and normalization  

### 2. Random Forest
- 500 trees with tuned hyperparameters  
- Feature importance tracking  
- Grid search optimization  

### 3. Decision Trees
- **Unpruned** (max depth) and **Pruned** (cross-validation for complexity parameter)  
- Demonstrates overfitting vs generalization  

### 4. Ensemble Method
- Weighted average of XGBoost + Random Forest  
- Grid search for optimal weights  
- Best performing overall (lowest RMSE)  

---

## ✅ Expected Output
- RMSE comparison of models  
- Best model selection  
- Holdout predictions saved to CSV  
- Decision tree visualizations  
- Feature importance insights  

---

## 📈 Model Performance
Models compared using RMSE:
- **XGBoost** – tuned gradient boosting  
- **Random Forest** – ensemble of trees  
- **Unpruned Decision Tree** – flexible, but overfits  
- **Pruned Decision Tree** – balanced complexity  
- **Ensemble** – combined top models, lowest RMSE  

**Result**: Ensemble method achieved the best predictive accuracy.  

---

## 🏠 Key Insights

### Important Predictors
- Living area square footage  
- Proximity to downtown Austin and UT  
- Property age (newer homes = higher prices)  
- Neighborhood-level characteristics (zipcode trends)  
- Quality features (garage, view, amenities)  

### Feature Engineering Impact
- **Interaction terms** capture complex relationships  
- **Neighborhood intelligence** adds local context  
- **Distance metrics** quantify location value  
- **Ratio features** capture efficiency of space usage  

---

## 📊 Holdout Predictions
- **Input**: `austinhouses_holdout.csv` (if available)  
- **Output**: `austinhouses_holdout_with_predictions.csv`  
- Uses the same feature engineering + ensemble pipeline  

---

## 🔍 Model Validation
- **Cross-validation**: 8-fold stratified, ensures robustness  
- **Holdout test**: 20% of data kept for unbiased evaluation  

---

## 💡 Technical Highlights
- **Leak prevention**: Neighborhood stats computed from training only  
- **Ensemble**: Performance-based weighting of models  
- **Scalable**: Efficient feature engineering and modular code  

---

## 🛠️ Future Improvements
- Incorporate external datasets (crime, walkability, economics)  
- Add time series methods for market cycles  
- Explore deep learning for richer feature interactions  
- Build real-time retraining pipeline  
