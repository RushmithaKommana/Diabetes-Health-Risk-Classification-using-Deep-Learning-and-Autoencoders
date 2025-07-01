Diabetes Health Risk Classification Using Deep Learning & Autoencoders
Overview
This project focuses on early detection of diabetes by classifying individuals as Non-Diabetic, Pre-Diabetic, or Diabetic using a real-world healthcare dataset from Kaggle. The primary challenge was handling severe class imbalance in the data, which was addressed using SMOTE (Synthetic Minority Over-sampling Technique).

Approach & Methodology
Performed data preprocessing, feature engineering, and class imbalance handling using SMOTE.

Built two models:

A Baseline Deep Neural Network (DNN)

An Autoencoder-Enhanced DNN that uses compressed feature representations for better minority class detection.

Conducted model evaluation using metrics like Accuracy, F1-Score, and AUC-ROC.

Performed SHAP value analysis for feature importance and model interpretability.

Compared model performance against traditional ML models like Logistic Regression, KNN, and XGBoost.

Tools & Technologies
Python, TensorFlow/Keras, Scikit-learn, SMOTE, SHAP, Pandas, NumPy, Google Colab

Key Outcomes
Achieved ~85% accuracy with Baseline DNN, with better minority class sensitivity using Autoencoder-DNN.

Improved fairness and balanced performance across all three classes (especially Pre-Diabetic and Diabetic categories).

Provided insights into important health indicators like BMI, General Health, and High Blood Pressure using SHAP analysis.

Challenges Faced
Severe class imbalance

Overfitting risks

Improving minority class recall

Future Work
Implement cost-sensitive learning, ensemble methods, and advanced SMOTE variants.

Explore transfer learning for better feature generalization.
