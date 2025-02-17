#region I. Import modules and cleaned dataset

import pandas as pd
pd.set_option('display.max_columns', None) # Show all columns
import numpy as np
np.random.seed(1)
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns

path = r"\data\dataset_train_clean.csv"
OG = pd.read_csv(path, encoding='latin-1', na_values=[])
OG = OG.fillna("None")

# creating our predicted variable and encoding it
OG["has_medal"] = OG["Medal"].apply(lambda x: 0 if x=="None" else 1)
# creating a new dummy to entail the home effect
OG["is_at_home"] = (OG["Country"] == OG["Host_country"]).astype(int)

#endregion

#region II. ML

# We will try to fit an appropriate model to best predict if an athlete wins a medal or not from this dataset

#region EDA

# List of continuous variables
continuous = list(OG.select_dtypes(include=[int, float]).columns)
continuous.remove("has_medal")
continuous.remove("ID") #ID is arbitrary so we remove it

# List of categorical variables
categorical = list(OG.select_dtypes(include=["object"]).columns)

# Plotting the heatmap from the correlation between continous variables matrix
corr_mat = OG[continuous].corr()
plt.figure(figsize=(6,5))
plt.title("Correlation matrix between continous variables in heatmap")
mat = sns.heatmap(corr_mat, annot=True, cmap="coolwarm")
plt.show()

# As observed in Data_viz, no strong (perfect) correlation is displayed except between height, weight and sex which is normal. Also the correlation Year and Sex is simply explained by the fact that more women took part to OG with time, hence more winning
# Therefor we assume that continous is not directly eliminated to best predict if an athlete wins a medal or not

#Displaying the frequency of winning a medal or not
freq = [i / len(OG) for i in OG["has_medal"].value_counts().to_list() ]
freq_w = freq[1]
freq_nw = freq[0]
print("Frequency of winning a medal is: ",freq_w)
print("Frequency of not winning a medal is: ",freq_nw)

freq_plot = sns.barplot(x=["0","1"],y=freq)
plt.title("Winning at the OG frequency")
plt.show()
for i in freq_plot.containers:
    freq_plot.bar_label(i)

# Target class 1 (winning a medal) represents ~15% of the dataset

#endregion

#region Data preprocessing

# Preprocessing tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder #LabelEncoder : we will only using TargetEncoder here since, whith our choice of columns,
#  OneHotEncoder would use 2.7GB of RAM, making it unsable for array types, necessary to XG_Boost regressions
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# The explanatory variables will be: ['Sex', 'Age', 'Height', 'Weight', 'Year', 'Season', 'is_at_home','Team, 'Games', 'Sport']

continuous_var = ["Age", "Height", "Weight", "Year"]
binary_var = ["Sex", "Season", "is_at_home"]  # already binary variables
remaining_categorical_features = ["Team", "Games", "Sport"]

X = OG[continuous_var + binary_var + remaining_categorical_features]
y = OG["has_medal"]

### Splitting the dataset between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # adding stratify=y to have an equal proportion of y between train and test


### Encoding Sex and Season in order to include them into the regression

X_train["Sex"] = X_train["Sex"].replace({'F': 0, 'M': 1}).astype(int)
X_train["Season"] = X_train["Season"].replace({"Summer":1,"Winter":0}).astype(int)
X_test["Sex"] = X_test["Sex"].replace({'F': 0, 'M': 1}).astype(int)
X_test["Season"] = X_test["Season"].replace({"Summer":1,"Winter":0}).astype(int)

### Standardizing continuous variables

std_scaler = StandardScaler()
X_train_cont = std_scaler.fit_transform(X_train[continuous_var])
X_test_cont = std_scaler.transform(X_test[continuous_var])

# Converting continuous to dataframe
X_train_cont_df = pd.DataFrame(X_train_cont, columns=continuous_var, index=X_train.index)
X_test_cont_df = pd.DataFrame(X_test_cont, columns=continuous_var, index=X_test.index)

### Encoding remaining categorical variables with OneHotEncoder

target_encoder = TargetEncoder()
X_train_cat = target_encoder.fit_transform(X_train[remaining_categorical_features], y_train)
X_test_cat = target_encoder.transform(X_test[remaining_categorical_features])

# Converting to dataframe
X_train_cat_df = pd.DataFrame(X_train_cat, columns=remaining_categorical_features, index=X_train.index)
X_test_cat_df = pd.DataFrame(X_test_cat, columns=remaining_categorical_features, index=X_test.index)

mean_encoded_value = X_train_cat_df.mean()
X_test_cat_df = X_test_cat_df.fillna(mean_encoded_value)

# Keeping at_home column
X_train_binary_df = X_train[binary_var].reset_index(drop=True)
X_test_binary_df = X_test[binary_var].reset_index(drop=True)

# Drop former categorical and continuous columns and concatenate into a new dataframe
X_train = X_train.drop(columns=remaining_categorical_features + continuous_var).reset_index(drop=True)
X_test = X_test.drop(columns=remaining_categorical_features + continuous_var).reset_index(drop=True)

X_train = pd.concat([X_train_binary_df, X_train_cont_df, X_train_cat_df], axis=1)
X_test = pd.concat([X_test_binary_df, X_test_cont_df, X_test_cat_df], axis=1)

#endregion

#region Model training and evaluation

### Training

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC # Support Vector Machine for classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB

# Performance metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve, classification_report
)

# We will not be using logistic regression since its too risky with TargetEncoder

#region --- Personal: Fixing index issues between X_train and y_train ---

# Train sets
print(f"X_train's shape : {X_train.shape}")
print(f"y_train's shape : {y_train.shape}")

print(f"y_train's original shape before transformation : {len(y)}")
print(f"Shape after transformation : {len(y_train)}")

print("X_train's first indexes before transformation :", X_train.index[:5])
print("y_train's first indexes before transformation :", y_train.index[:5])

print(f"Number of lignes lost after transformation : {271116 - 216892}")

# Checking which lines still remain in X_train
y_train_realigned = y_train.loc[y_train.index.intersection(X_train.index)]
print(f"y_train new shape after realignment : {y_train_realigned.shape[0]}")
X_train = X_train.loc[y_train_realigned.index]
print(f"X_train shape after correction : {X_train.shape}")
print(f"y_train shape after correction : {y_train_realigned.shape}")
y_train = y_train_realigned

print(f"X_train final shape: {X_train.shape}")
print(f"y_train final shape : {y_train.shape}")
print(f"X_train's first indexes : {X_train.index[:5]}")
print(f"y_train's first indexes : {y_train.index[:5]}")

# Test sets
# Vérifier les indices communs entre X_test et y_test
common_indices = X_test.index.intersection(y_test.index)

# Réaligner X_test et y_test pour ne garder que les indices communs
X_test = X_test.loc[common_indices]
y_test = y_test.loc[common_indices]

assert X_test.shape[0] == y_test.shape[0], "Erreur : X_test et y_test ne sont pas alignés !"

# Check after correction
print(f" Shape after correction : X_test={X_test.shape}, y_test={y_test.shape}")


#endregion

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

randomforest = RandomForestClassifier()
randomforest.fit(X_train,y_train)

XG_Boost =XGBClassifier()
XG_Boost.fit(X_train,y_train)

### Evaluation

tree_pred= tree.predict(X_test)
forest_pred = randomforest.predict(X_test)
XGB_pred = XG_Boost.predict(X_test)

print("Decision Tree:")
print("accuracy:", accuracy_score(y_true=y_test, y_pred=tree_pred))
print("f1 score:", f1_score(y_true=y_test, y_pred=tree_pred))

print("Random Forest:")
print("accuracy:", accuracy_score(y_true=y_test, y_pred=forest_pred))
print("f1 score:", f1_score(y_true=y_test, y_pred=forest_pred))

print("XGBoost:")
print("accuracy:", accuracy_score(y_true=y_test, y_pred=XGB_pred))
print("f1 score:", f1_score(y_true=y_test, y_pred=XGB_pred))

# Confusion matrixes

classes = [0, 1]

confusion_mat_tree = confusion_matrix(y_test, tree_pred, labels=classes, normalize="true")
confusion_mat_tree
confusion_mat_forest = confusion_matrix(y_test, forest_pred, labels=classes, normalize="true")
confusion_mat_forest
confusion_mat_XGB = confusion_matrix(y_test, XGB_pred, labels=classes, normalize="true")
confusion_mat_XGB

disp_tree = ConfusionMatrixDisplay(confusion_matrix=confusion_mat_tree, display_labels=classes)
disp_tree.plot()
plt.title("Confusion Matrix: Decision Tree")
plt.show()
disp_forest = ConfusionMatrixDisplay(confusion_matrix=confusion_mat_forest, display_labels=classes)
disp_forest.plot()
plt.title("Confusion Matrix: Random Forest")
plt.show()
disp_XGB = ConfusionMatrixDisplay(confusion_matrix=confusion_mat_XGB, display_labels=classes)
disp_XGB.plot()
plt.title("Confusion Matrix: XGBoost")
plt.show()


# Additional metrics
metrics = {}
models = {
    "Decision Tree": tree_pred,
    "Random Forest": forest_pred,
    "XGBoost": XGB_pred
}

for model_name, y_pred in models.items():
    metrics[model_name] = {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_pred)
    }

# Displaying results
for model, scores in metrics.items():
    print(f"\n🔹 {model}:")
    print(f"Precision: {scores['Precision']:.4f}")
    print(f"Recall: {scores['Recall']:.4f}")
    print(f"AUC: {scores['AUC']:.4f}")

# ROC AUC curve
plt.figure(figsize=(8, 6))

for model_name, y_pred in models.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC AUC Curve")
plt.legend()
plt.show()

# From this curve, it seems that Decision tree is slightly more performant than other models when it comes to split classes

# Precision-Recall curve
plt.figure(figsize=(8, 6))

for model_name, y_pred in models.items():
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred)
    plt.plot(recall_curve, precision_curve, label=model_name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# From this curve, it seems that Random Forest has the best Precision-Recall trade-off

#region Finding hyperparameters for XG Boost

from sklearn.model_selection import GridSearchCV
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Class balanceness parameter

# Defining hyperparameters to test
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'gamma': [0, 0.1, 0.2],  # Avoiding overfitting
    'subsample': [0.6, 0.8, 1.0],  # Sampling
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, scale_pos_weight],  # Dealing with classes unbalanceness
}

# Initializing XGBoost

xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss')

# Initializing GridSearchCV

grid_xgb = GridSearchCV(
    xgb_model, param_grid=param_grid_xgb,
    cv=3, scoring='f1', verbose=1, n_jobs=-1  # n_jobs=-1 pour utiliser tous les cœurs CPU
)

# Training GridSearchCV
grid_xgb.fit(X_train, y_train)

# Displaying better parameters and F1 score
print("Best hyperparameters found :", grid_xgb.best_params_)
print("Best F1 score obtained :", grid_xgb.best_score_)

# Get the model/estimator with the optimal hyperparameters
best_model = grid_xgb.best_estimator_

# Predict values for X_test using the model
best_model_predict = grid_xgb.predict(X_test)
best_model_f1 = f1_score(y_true=y_test, y_pred=best_model_predict)
print("f1 score test:", best_model_f1)

#endregion

#endregion

#endregion

#region III. Explainability with shap

import shap
np.bool=bool

df_shap = X_train.iloc[:100]

explainer = shap.TreeExplainer(randomforest)
shap_values = explainer.shap_values(df_shap)

# Explainability bar plot
shap.summary_plot(shap_values[:,:,0], feature_names=df_shap.columns, plot_type="bar")

# Other explainability bar plot
shap.summary_plot(shap_values[:,:,0], features=df_shap, feature_names=df_shap.columns, plot_type="dot")


# Key takeaways from the analysis:

# - The model is highly influenced by is_at_home.
# - Some variables have very low impact (Age, Weight, Sex), which seems strange especially for age.
# - The model heavily relies on Team and Sport.

# Conclusion: we should try other models or take a step back to data preprocessing


#endregion

#region IV. DL

from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLu

def nn_model(nb_features, nb_classes):
    model = Sequential()
    model.add(Dense(128, input_dim=nb_features, activation='relu'))  # Plus de neurones
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(nb_classes, activation='sigmoid'))  # Softmax → Sigmoid si binaire
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = nn_model(nb_features=X_train.shape[1], nb_classes=1)
model.summary()

model.fit(X_train, y_train, epochs=8, batch_size=10, validation_split=0.2, verbose=1)

history = model.history

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.show()

# Courbe d'accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.show()


#endregion