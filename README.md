# **Telecom Customer Churn Prediction**

# Overview
This project aims to develop a machine learning model to predict whether a customer will soon stop doing business with SyriaTel, a telecommunications company. The problem at hand is a binary classification task where the goal is to identify customers who are likely to churn, i.e., discontinue their subscriptions or services with the company.

# Background

Customer churn, or customer attrition, poses a significant challenge for telecommunications companies. Losing customers not only impacts revenue but also reflects underlying issues with service quality, customer satisfaction, and competitive positioning. By accurately predicting churn, telecom companies can proactively address customer concerns, improve retention strategies, and enhance overall customer satisfaction.

# Dataset
The dataset used for this project is sourced from Kaggle: [Link to the dataset](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset). It contains various features such as:

**state** The state where the customer resides (categorical)

**account length** The number of days the customer has been with the company (numerical)

**area code** The area code of the customer's phone number (categorical)

**phone number** The customer's phone number (categorical)

**international plan** Whether the customer has an international calling plan (categorical)

**voice mail plan** Whether the customer has a voicemail plan (categorical)

**number vmail messages** Number of voicemail messages (numerical)

**total day minutes** Total minutes of daytime calls (numerical)

**total day calls** Total number of daytime calls (numerical)

**total day charge** Total charge for daytime calls (numerical)

**total eve minutes** Total minutes of evening calls (numerical)

**total eve calls** Total number of evening calls (numerical)

**total eve charge** Total charge for evening calls (numerical)

**total night minutes** Total minutes of nighttime calls (numerical)

**total night calls** Total number of nighttime calls (numerical)

**total night charge** Total charge for nighttime calls (numerical)

**total intl minutes** Total minutes of international calls (numerical)

**total intl calls** Total number of international calls (numerical)

**total intl charge** Total charge for international calls (numerical)

**customer service calls** Number of customer service calls made (numerical)

**churn**  Whether the customer churned (i.e., stopped doing business with the company). The target variable is binary, indicating whether a customer churned or not.

# Objective
The primary objective of this project is to build a predictive model that can effectively identify customers at risk of churn. By leveraging machine learning algorithms and analyzing historical customer data, the aim is to develop a robust model capable of accurately predicting churn, thereby enabling SyriaTel to implement targeted retention strategies and mitigate customer attrition.

# Approach
**Data Exploration** Conduct exploratory data analysis (EDA) to understand the characteristics and distributions of the dataset features, identify patterns, and gain insights into factors influencing customer churn.

**Feature Engineering** Preprocess the data and engineer relevant features that may improve the predictive performance of the model. This involves handling missing values, encoding categorical variables, scaling numerical features, and creating new features as needed.

**Model Development** Train and evaluate machine learning models using appropriate algorithms such as logistic regression, decision trees, random forests, gradient boosting, etc. Evaluate model performance using suitable metrics and techniques such as cross-validation and hyperparameter tuning.


# Technologies Used
1.Python

2.Pandas, NumPy, Scikit-learn for data manipulation and modeling

3.Jupyter Notebook for exploratory data analysis and model development

4.GitHub for version control 

# Visualizations 

**correlation matrix of numerical variables**

![Correlation Matrix](https://github.com/liciemw/Phase3-Project/blob/master/Correlation%20matrix.png)


**Performance Metrics for KNN**

![Knn Performance Matrices](https://github.com/liciemw/Phase3-Project/blob/master/Knn%20performance%20matrices.png)


The model correctly predicts the class label for about 88.2% of the instances in the testing dataset. The classification report provides a detailed summary of performance metrics for each class 0(customers not likely to churn) and 1(customers likely to churn), including precision, recall, and F1-score Precision: Precision measures the proportion of true positive predictions among all positive predictions made by the model. In other words, it measures the accuracy of positive predictions. A precision score of 0.88 for class 0 means that 88% of the instances predicted as class 0 were actually class 0. Similarly, a precision score of 0.89 for class 1 means that 89% of the instances predicted as class 1 were actually class 1.

Recall: Recall measures the proportion of true positive predictions among all actual positive instances in the dataset. It quantifies the ability of the model to correctly identify positive instances. A recall score of 0.99 for class 0 means that 99% of the actual instances of class 0 were correctly classified by the model. Conversely, a recall score of 0.25 for class 1 indicates that only 25% of the actual instances of class 1 were correctly classified.

F1-score: The F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall, considering both false positives and false negatives. A higher F1-score indicates better overall performance. The F1-score for class 0 is 0.93, indicating a good balance between precision and recall, while the F1-score for class 1 is lower at 0.39, reflecting the trade-off between precision and recall for this class.

Support: Support represents the number of actual occurrences of each class in the testing dataset. It provides context for interpreting precision, recall, and F1-score metrics.

**Performance Metrics for Random Forest Classifier**

![RF Performance Matrices](https://github.com/liciemw/Phase3-Project/blob/master/RF%20performance%20matrices.png)

The precision score of 0.9516 indicates that 95.16% of the instances predicted as churn were actually churn.

The recall score of 0.5842 means that the model correctly identified 58.42% of the actual churn instances.

The Random Forest classifier achieves a higher accuracy of 93%, indicating that it effectively discriminates between churn and non-churn instances. However, the recall score suggests that there is room for improvement in correctly identifying all churn instances.

**metrics performance of the tuned model**

![Tuned Matrices](https://github.com/liciemw/Phase3-Project/blob/master/tuned%20matrices.png)


The XGBoost tuned model performs quite well on both the training and testing datasets, with an accuracy of approximately 98.46% on the training dataset and 95.80% on the testing dataset. Having high accuracy on both datasets suggests that the model generalizes well to unseen data, indicating that it has learned meaningful patterns from the training data and can make accurate predictions on new data points. Visualizing the other evaluation metrics such as precision, recall of the tuned model.

# Conclusion
We have successfully improved the all the metrics values therefore an improvement in correctly identifying all churn instances. The final tuned model does incredibly well in its ability to correctly classify positive and negative instances and balance between false positives and false negatives. In conclusion, the developed model provides valuable insights for SyriaTel to proactively identify and retain customers at risk of churning. By leveraging predictive analytics, telecom companies can optimize customer retention strategies, enhance customer satisfaction, and minimize revenue loss.



```python

```
