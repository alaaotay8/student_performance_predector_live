import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
data = pd.read_csv('Student_performance_data _.csv')

# Examine data
print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Missing values: {data.isnull().sum().sum()}")

# Feature Engineering
print("Performing advanced feature engineering...")

# Create new features
data['StudyEfficiency'] = data['StudyTimeWeekly'] / (data['Absences'] + 1)  # Study time per absence
data['SupportIndex'] = data['ParentalSupport'] + data['Tutoring']  # Combined support
data['ActivityScore'] = data['Extracurricular'] + data['Sports'] + data['Music'] + data['Volunteering']
data['StudyIntensity'] = pd.cut(data['StudyTimeWeekly'], bins=3, labels=[0, 1, 2]).astype(int)
data['AbsenceLevel'] = pd.cut(data['Absences'], bins=3, labels=[0, 1, 2]).astype(int)

# Age groups
data['AgeGroup'] = pd.cut(data['Age'], bins=[14, 16, 18, 20], labels=[0, 1, 2]).astype(int)

# Additional advanced features
data['StudyPerAge'] = data['StudyTimeWeekly'] / data['Age']
data['SupportPerActivity'] = data['SupportIndex'] / (data['ActivityScore'] + 1)
data['EducationSupport'] = data['ParentalEducation'] * data['SupportIndex']

# Interaction features
data['TutoringXSupport'] = data['Tutoring'] * data['ParentalSupport']
data['StudyXSupport'] = data['StudyTimeWeekly'] * data['SupportIndex']
data['ActivityXEducation'] = data['ActivityScore'] * data['ParentalEducation']

# Prepare data
feature_cols = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 
                'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 
                'Sports', 'Music', 'Volunteering', 'StudyEfficiency', 'SupportIndex',
                'ActivityScore', 'StudyIntensity', 'AbsenceLevel', 'AgeGroup',
                'StudyPerAge', 'SupportPerActivity', 'EducationSupport',
                'TutoringXSupport', 'StudyXSupport', 'ActivityXEducation']

X = data[feature_cols]
y = data['GradeClass']

print(f"Input features ({len(feature_cols)}): {X.columns.tolist()}")
print(f"Target class distribution: {y.value_counts().sort_index()}")
print(f"Class imbalance ratio: {y.value_counts().max() / y.value_counts().min():.2f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# Apply SMOTE to handle class imbalance
print("Applying SMOTE for class balancing...")
smote = SMOTE(random_state=123, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Training set shape: {X_train_balanced.shape}")
print(f"After SMOTE - Class distribution: {pd.Series(y_train_balanced).value_counts().sort_index()}")

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Define improved models
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=300, 
        max_depth=20, 
        min_samples_split=5, 
        min_samples_leaf=2, 
        class_weight='balanced', 
        random_state=123
    ),
    'Extra Trees': ExtraTreesClassifier(
        n_estimators=300, 
        max_depth=20, 
        min_samples_split=5,
        min_samples_leaf=2, 
        class_weight='balanced', 
        random_state=123
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=8, 
        min_samples_split=10, 
        min_samples_leaf=4, 
        random_state=123
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=8, 
        subsample=0.8,
        colsample_bytree=0.8, 
        random_state=123, 
        eval_metric='mlogloss'
    ),
    'Logistic Regression': LogisticRegression(
        random_state=123, 
        max_iter=2000, 
        class_weight='balanced', 
        C=0.1
    ),
    'SVM': SVC(
        random_state=123, 
        class_weight='balanced', 
        kernel='rbf', 
        C=10.0, 
        gamma='scale'
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=11, 
        weights='distance'
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=123, 
        max_depth=15, 
        min_samples_split=10, 
        min_samples_leaf=4, 
        class_weight='balanced'
    ),
    'AdaBoost': AdaBoostClassifier(
        random_state=123, 
        n_estimators=200, 
        learning_rate=0.5
    ),
    'Naive Bayes': GaussianNB()
}

print("\nComparing improved models...")
print("="*60)

best_score = 0
best_model_name = ""
best_model = None
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for models that need it
    if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors', 'Naive Bayes']:
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
        y_train_model = y_train_balanced
    else:
        X_train_model = X_train_balanced
        X_test_model = X_test
        y_train_model = y_train_balanced
    
    # Training
    model.fit(X_train_model, y_train_model)
    
    # Prediction
    y_pred = model.predict(X_test_model)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation on original data to get realistic performance
    if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors', 'Naive Bayes']:
        cv_X = scaler.fit_transform(X_train)
        cv_scores = cross_val_score(model, cv_X, y_train, cv=5, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'Accuracy': accuracy,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    }
    
    print(f"{name}:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Cross-validation Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("-" * 40)
    
    # Determine best model based on test accuracy
    if accuracy > best_score:
        best_score = accuracy
        best_model_name = name
        best_model = model
        best_X_train = X_train_model
        best_X_test = X_test_model

print(f"\nBest model: {best_model_name}")
print(f"Best model test accuracy: {best_score:.4f}")

# Feature importance for tree-based models
if best_model_name in ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'XGBoost', 'Decision Tree']:
    print(f"\nTop 10 Feature Importances for {best_model_name}:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, (feature, importance) in enumerate(feature_importance.head(10).values):
        print(f"{i+1:2d}. {feature:20s}: {importance:.4f}")

# Train best model on full balanced data and save it
print(f"\nTraining and saving best model ({best_model_name}) on full dataset...")

# Retrain best model on full balanced data
if best_model_name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors', 'Naive Bayes']:
    X_full_balanced, y_full_balanced = smote.fit_resample(X, y)
    X_full_scaled = scaler.fit_transform(X_full_balanced)
    best_model.fit(X_full_scaled, y_full_balanced)
    # Save model with scaler
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'smote': smote,
        'feature_names': X.columns.tolist(),
        'model_name': best_model_name,
        'requires_scaling': True,
        'uses_smote': True
    }
else:
    X_full_balanced, y_full_balanced = smote.fit_resample(X, y)
    best_model.fit(X_full_balanced, y_full_balanced)
    # Save model without scaler
    model_data = {
        'model': best_model,
        'scaler': scaler,  # Still save scaler for consistency
        'smote': smote,
        'feature_names': X.columns.tolist(),
        'model_name': best_model_name,
        'requires_scaling': False,
        'uses_smote': True
    }

# Save model
joblib.dump(model_data, 'enhanced_student_perf_model.pkl')

# Print detailed report for best model
print(f"\nDetailed Classification Report for {best_model_name}:")
print("="*60)
y_pred_best = best_model.predict(best_X_test)
print(classification_report(y_test, y_pred_best))

# Confusion Matrix
print(f"\nConfusion Matrix for {best_model_name}:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

print(f"\nModel performance summary:")
print(f"- Best model: {best_model_name}")
print(f"- Test accuracy: {best_score:.4f}")
print(f"- Features used: {len(feature_cols)}")
print(f"- Training samples after SMOTE: {len(y_full_balanced)}")

print("\nModel saved successfully to: enhanced_student_perf_model.pkl")
print("Model is ready for use!")

# Save results to CSV for analysis
results_df = pd.DataFrame(results).T
results_df.to_csv('model_comparison_results.csv')
print("Model comparison results saved to: model_comparison_results.csv")