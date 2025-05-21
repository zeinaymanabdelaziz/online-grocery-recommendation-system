# Online Grocery Recommendation System

A machine learning project designed to predict whether users will reorder grocery products based on their past behavior.  
<br>Developed as part of an **Introduction to Data Science** course using Python, Scikit-learn, and Jupyter Notebook.

# Project Overview

This project analyzes customer purchase patterns to build a predictive model that recommends products for reordering.  
<br>We evaluated and compared three machine learning models:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Logistic Regression

The best-performing model was selected based on multiple evaluation metrics.

# Key Features

- Analyzed merged user order and product datasets
- Engineered features capturing user/product behavior
- Compared classification models using Scikit-learn
- Evaluated using Accuracy, Precision, Recall, and AUC-ROC
- Recommended the most effective model for real-world deployment

# Dataset Features

Key columns used for training:

- `user_total_orders`
- `user_total_products`
- `user_reorder_ratio`
- `product_total_orders`
- `product_reorder_ratio`
- `add_to_cart_order`
- `order_dow`
- `order_hour_of_day`
- `days_since_prior_order`

**Target Variable:**  
- `reordered` (Binary: 1 if product was reordered, 0 otherwise)

# Model Comparison Results

| Model              | Accuracy | Precision | Recall | AUC-ROC |
|--------------------|----------|-----------|--------|---------|
| KNN (k=3)          | 0.66     | 0.70      | 0.76   | 0.69    |
| Decision Tree      | 0.66     | 0.71      | 0.71   | 0.65    |
| Logistic Regression| 0.66     | 0.70      | 0.75   | 0.70    |

✅ **Selected Model:** Logistic Regression  
Chosen for its strong balance of metrics and efficiency during prediction.

# Project Structure

📦 Grocery_Recommender
<br>├── Final_Project_Code.ipynb # Jupyter notebook with data analysis and modeling
<br>├── Final_Project_Report.pdf # Full report documenting methodology and results


# Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Jupyter Notebook
- Matplotlib / Seaborn (optional for visualization)

# Learning Outcomes

- Practical use of machine learning models on real-world datasets
- Feature engineering and preprocessing
- Model evaluation and comparison
- Hands-on Jupyter-based experimentation

# Future Work

- Try ensemble models (e.g., Random Forest, Gradient Boosting)
- Optimize hyperparameters using GridSearchCV
- Incorporate more user-product interaction features
- Deploy as a web service or integrate into retail platforms

# License

This project is for academic and educational purposes only.
