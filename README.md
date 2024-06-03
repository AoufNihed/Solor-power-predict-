Solar Power Energy Generation Prediction
This project aims to predict solar power energy generation using machine learning models. The dataset used in this project contains various environmental features like radiation, air temperature, wind speed, etc., and the target variable is the solar power energy generation.

Dataset Description
The dataset used in this project contains the following columns:

Radiation: The amount of solar radiation.
AirTemperature: The temperature of the air.
WindSpeed: The speed of the wind.
RelativeAirHumidity: The relative humidity of the air.
SystemProduction: The generated solar power energy.
Exploratory Data Analysis (EDA)
Correlation Analysis
A correlation matrix was computed to understand the relationships between different features and the target variable. The heatmap visualization of the correlation matrix is shown.

Seasonal Trends
Seasonal decomposition analysis was performed to analyze the trends in solar power energy generation over different seasons.

Hourly Analysis
Hourly analysis was conducted to analyze solar power energy generation trends throughout the day.

Data Preprocessing
Feature Engineering
The date-time column was split into separate columns for day, month, year, and hour.
Seasonal labels were assigned based on the month column.
Categorical Encoding
Ordinal encoding was applied to categorical variables like season, time, and humidity.

Data Splitting
The dataset was split into training and testing sets for model training and evaluation.

Model Building
Linear Regression
A linear regression model was built using the OLS method.

Random Forest Regression
A random forest regression model was trained and evaluated.

Model Evaluation
Confusion Matrix
Confusion matrices were generated for the following classification models:

Logistic Regression
Support Vector Machine (SVM)
Gradient Boosting
Conclusion
In this project, we explored various machine learning models to predict solar power energy generation based on environmental factors. The random forest regression model showed promising results in predicting solar power energy generation.
