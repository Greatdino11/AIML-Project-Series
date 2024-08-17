# Disease Prediction System using Machine Learning
# Using the Disease Symptoms and Patient Profile Dataset from Kaggle

# 1. Install Required Libraries (run this in your terminal)
# pip install pandas numpy scikit-learn matplotlib seaborn shap

# 2. Data Collection
import pandas as pd

# Load the dataset
data = pd.read_csv('data.csv')

# Display the first few rows of the dataset
print(data.head())

# 3. Data Preprocessing
# Handle missing values (if any)
data = data.dropna()

# Encode categorical variables
data['Disease'] = data['Disease'].astype('category').cat.codes
data['Gender'] = data['Gender'].astype('category').cat.codes

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = data.drop(['Outcome Variable'], axis=1)  # Drop the target column ('Outcome Variable')
X_scaled = scaler.fit_transform(X)
y = data['Outcome Variable']  # Define target variable

# 4. Feature Selection
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation Analysis
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Feature Importance using RandomForest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_

# Plot feature importances
feat_importances = pd.Series(importances, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# 5. Model Development
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Implement Various Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC()
}

# Train and Evaluate Models
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f'{name}: {score}')

# 6. Cross-Validation
from sklearn.model_selection import cross_val_score

# Cross-Validation for Random Forest
scores = cross_val_score(RandomForestClassifier(), X_scaled, y, cv=5)
print('Cross-Validation Scores:', scores)
print('Average Score:', scores.mean())

# 7. Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print('Best Parameters:', grid_search.best_params_)

# 8. Model Interpretability (Optional)
import shap

explainer = shap.TreeExplainer(grid_search.best_estimator_)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# 9. Validation and Testing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate Final Model
y_pred = grid_search.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred)}')
