# Introduction

Demand forecasting plays a crucial role in supply chain management and inventory optimization. Accurate demand predictions enable businesses to maintain optimal stock levels, reduce costs, and improve customer satisfaction. In this case study, we developed a forecasting model to predict Walmart sales across different stores and product categories for the next 28 days.

# Proposed Approach

The approach consisted of the following key steps:

## Data Exploration and Preprocessing

Before building any forecasting model, it is essential to explore and preprocess the data. The dataset contains multiple CSV files, including sales data, calendar events, and price information. The plan included:

* Inspecting the data to understand its structure and contents.

* Merging different datasets to create a comprehensive feature set.

* Transforming categorical variables, such as product categories and store locations, into numerical representations.

Although the file `process.py` was provided to perform basic preprocessing on the training dataset, we also created `process_test.py` to handle preprocessing for the test set. These files are available in the root of the repository.

## Feature Engineering

To enhance the predictive power of the approach, several meaningful features was created, such as:

* Time-based features (e.g., day of the week, holidays, and seasonality patterns).

* Lag features to capture temporal dependencies.

* Aggregated sales trends at the store and category levels.

While many other features could be created to further enrich the model, for educational purposes, we focused on the most relevant and straightforward ones to implement. Assistance from a business stakeholder could be valuable here to help identify additional variables and tags that directly impact sales and purchasing behavior, such as promotions, local events, or weather variations.

All these transformations were executed in the Feature Engineering section of the notebook modeling.ipynb.

## Model Selection and Training

Given the limited time available and the goal of quickly delivering a functional solution, a single model was focused on for forecasting sales across all stores and products. Although we could have explored multiple models, such as one for each product category, store, or region, the generic approach would be both efficient and scalable.

### Why a Single Model?
This decision was made for the following reasons:

1. Time Constraints: Developing separate models for each product category or store would have been time-consuming. A single model allows for a quicker implementation, which is crucial in a time-limited scenario like this one.

2. Maintainability: In a production environment, managing and deploying one model is much easier than handling multiple models for different stores or categories. A single model simplifies deployment and monitoring, making it easier to update and maintain over time.

3. Scalability: This approach allows for easier future expansion. As new products or stores are added, they can be incorporated into the existing model rather than needing separate models for each new addition.

4. Forecast Evaluation

### Why XGBoost?
I chose XGBoostRegressor as the model for this forecasting task due to several factors:

1. Computational Efficiency: XGBoost is highly efficient in handling large datasets and is much faster than deep learning models such as LSTM, especially for tabular data. Given the time constraints, it was essential to choose a model that could process the data quickly and deliver results efficiently.

2. Hyperparameter Tuning: XGBoost is known for requiring relatively less time to tune compared to deep learning models like LSTM. While LSTMs require extensive training and fine-tuning, XGBoost allows for a quicker optimization process, which was crucial for this project.

3. Minimal Preprocessing: XGBoost has built-in support for handling categorical features directly using the enable_categorical parameter. This eliminates the need for complex preprocessing steps like One-Hot Encoding, saving time and effort during the data transformation process.

4. Effectiveness for the MVP: For an MVP, the goal was to create a working solution that could be refined later. XGBoost is a robust and well-established model that can perform well with minimal data preprocessing and is known for its performance in forecasting tasks.

### Offline Performance
Model performance was assessed using the Mean Absolute Error (MAE), which provides a clear interpretation of forecast accuracy in the same units as the sales data. MAE was chosen because it is straightforward and easy to interpret, offering an intuitive measure of the average magnitude of forecast errors without being overly sensitive to outliers.

While I focused on MAE for simplicity and clarity in this project, other metrics such as Root Mean Squared Error (RMSE) and Weighted Scaled Pinball Loss (WSPL) could have been used as well. RMSE, for example, would be useful for measuring the magnitude of errors while penalizing larger deviations more heavily, which can be valuable in certain forecasting scenarios. WSPL, on the other hand, would be effective for evaluating quantile forecasts, providing insights into how well the model performs across different parts of the distribution.

The choice of MAE reflects the goal of balancing simplicity and interpretability in the evaluation process for this MVP, but these alternative metrics could provide additional perspectives on model performance if needed in the future.

### Online Performance
While offline metrics like MAE offer valuable insights into how well the model performs on historical data, online performance monitoring focuses on how well the model adapts to new, unseen data over time and in production environments.


* Real-Time Forecast Accuracy: Unlike offline evaluation, which uses static datasets, online performance requires constant monitoring of forecast accuracy in real-time. New sales data will be collected continuously, and the model's ability to predict demand for each day will need to be tracked as the model is exposed to new patterns that may not have been present during training.

* Data Drift: One of the primary concerns in online monitoring is data drift—when the underlying patterns in the data change over time. This could happen due to seasonality shifts, changes in consumer behavior, or other market factors. Monitoring will involve checking if the model’s predictions are still in line with actual sales data and whether there are any signs of performance degradation.

* Model Recalibration: To maintain high forecasting accuracy in the face of data drift, the model may need to be periodically retrained using the most recent data. Retraining ensures that the model adapts to new trends and seasonality patterns, improving its long-term accuracy.

* Business Metrics: In addition to forecast accuracy, business performance metrics like inventory levels, fulfillment rates, and stockouts will be closely monitored to assess how well the model’s predictions are translating into real-world business outcomes. If the model's forecasts result in suboptimal stock levels, further adjustments to the model might be necessary.

* Alerting and Monitoring Systems: In a live setting, automated alerting systems can be set up to notify stakeholders if forecast accuracy drops below a certain threshold. This will help detect issues early on and allow for swift intervention to correct the model's behavior if necessary.

* A/B Testing: To continuously improve performance, A/B testing can be employed, where different versions of the model are tested simultaneously in production. This allows for the comparison of model performance under real-world conditions and helps identify the best-performing model over time.

# Results

The proposed approach achieved a Mean Absolute Error (MAE) of 1.0891 on the test data – see the Predicting section in `modeling.ipynb`. It is important to note that the MAE tends to be lower for the earlier days of prediction and gradually increases as the forecast horizon extends. This is a common behavior in time series forecasting, as the model is typically more accurate for near-term predictions, which are closer to historical patterns.

To benchmark the performance, we've created a simple baseline model based on the average prices per week for each item-store pair. This baseline model, implemented in `baseline.ipynb`, resulted in a MAE of 1.1315. By using XGBoost, our approach achieved an uplift of approximately 3.75% in forecast accuracy compared to the baseline.

While this 3.75% improvement in accuracy might seem modest, it highlights the potential for significant financial returns when forecasting demand in a large-scale retail operation like Walmart. Even small improvements in model performance can lead to large impacts, as more accurate forecasts help better match supply with demand, reduce stockouts, and optimize inventory management. Translating this uplift into tangible business benefits would require collaboration with stakeholders to estimate the potential savings or revenue increase, but given the scale of Walmart’s operations, even marginal improvements in forecasting accuracy can translate into substantial financial gains.

# Structure
The project structure is as follows:

* `process_test.py`: Handles the preprocessing of the test dataset.

* Notebooks:
    * `baseline.ipynb`: Development, usage, and evaluation of the baseline model.
    * `modeling.ipynb`: Training, prediction, and evaluation of the XGBoost model.

* Submission: 
    * `predictions.csv`: Contains the predictions for the 28-day period in CSV format, as requested.