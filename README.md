# Analyzing Second-Hand Car Sales Data with Supervised and Unsupervised Learning Models
Abstract
This project applies various machine learning techniques to analyze second-hand car sales data in the UK. The objectives are to predict car prices using regression models and identify patterns through clustering. Models such as Linear Regression, Polynomial Regression, Multiple Variable Regression, Random Forest Regression, and an Artificial Neural Network (ANN) are implemented for price prediction. K-means clustering is used to identify patterns within the data. Among the models, the Random Forest Regressor achieved the highest R² score of 0.9985. The study highlights the importance of feature selection and model complexity in achieving high prediction accuracy.

# Introduction
The project explores the use of machine learning to predict second-hand car prices and uncover patterns in the dataset. The dataset consists of 55,000 records, with features such as manufacturer, model, engine size, fuel type, year of manufacture, mileage, and price. Both supervised learning models (regression) and unsupervised learning models (clustering) are implemented. Models are evaluated using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) score.

# Methodology
Tools and Libraries
Languages: Python
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Keras
Environment: Jupyter Notebook
Data Preparation
The dataset was preprocessed by handling missing values and ensuring correct data types. Features such as 'Manufacturer', 'Model', 'Engine size', 'Fuel type', 'Year of manufacture', 'Mileage', and 'Price' were used for analysis. Numerical features were standardized, and categorical features were one-hot encoded where necessary.

Supervised Learning Models
Linear Regression
Used to predict car prices based on individual features ('Engine size', 'Year of manufacture', 'Mileage'). Performance was measured using MAE, MSE, and R² scores.

Polynomial Regression
Polynomial features of degrees 2, 3, and 4 were generated to capture non-linear relationships in the data. The performance was compared using the same evaluation metrics.

Multiple Variable Regression
This model used multiple numerical features ('Engine size', 'Year of manufacture', 'Mileage') to predict car prices. It showed better performance than the individual linear regression models.

Random Forest Regression
Random Forest was applied using both numerical and categorical features. The model achieved the highest accuracy (R² score of 0.9985).

Artificial Neural Network (ANN)
An ANN was developed to predict car prices using both numerical and categorical features. The model used two hidden layers with 64 neurons each, ReLU activation, and dropout layers for regularization. Early stopping was implemented to prevent overfitting.

Unsupervised Learning Models
k-Means Clustering
K-means clustering was applied to identify patterns in the dataset. The elbow method was used to determine the optimal number of clusters, and results were evaluated using the silhouette score and Davies-Bouldin Index.

Agglomerative Clustering
Agglomerative clustering was compared with k-means, using complete linkage and three clusters. Results showed similar performance, with slightly lower silhouette scores.

# Results
Regression Model Performance
Model	Feature(s)	MAE	MSE	R² Score
Linear Regression	Engine size	10970.08	228864284.82	0.1398
Polynomial Regression	Year of manufacture (d=3)	5185.92	97708693.61	0.6327
Multiple Variable	Engine size, Year, Mileage	6167.89	84405262.36	0.6828
Random Forest	All numerical & categorical	282.16	386922.24	0.9985
Artificial Neural Network	All numerical & categorical	987.10	1032000.00	0.9847
Clustering Performance
Algorithm	Parameters	Silhouette Score	Davies-Bouldin Index
k-means Clustering	n_clusters=4	0.3496	0.9492
k-means (subset of features)	n_clusters=3	0.4629	0.7590
Agglomerative Clustering	n_clusters=3, linkage=complete	0.3333	0.9856
# Discussion
Linear and Polynomial Regression: Polynomial models captured non-linear relationships better than linear models, with R² scores of up to 0.6327.
Random Forest: The best-performing model, leveraging both numerical and categorical features, with an R² score of 0.9985.
Artificial Neural Network: The ANN model performed well with an R² score of 0.9847, highlighting the potential of deep learning in price prediction.
Clustering: K-means clustering identified market segments with a silhouette score of 0.4629 for specific features, indicating meaningful groupings in the data.
Limitations and Future Work
Feature Engineering: The study could benefit from incorporating more features, such as additional car specifications and time-based data.
Advanced Models: Future work could explore hybrid models, more advanced neural networks, and further clustering techniques.
Hyperparameter Tuning: More extensive hyperparameter optimization could improve the performance of regression and clustering models.
# Conclusion
This project demonstrates the effectiveness of machine learning models in predicting second-hand car prices and clustering patterns. The Random Forest Regressor and ANN achieved high accuracy, while k-means clustering revealed potential market segments. Feature selection and model complexity were crucial in achieving these results.