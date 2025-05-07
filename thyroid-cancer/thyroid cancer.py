import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Import models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

path = r"Thyroid_Diff.csv" #path of data
df = pd.read_csv(path)

print("Dataset Overview")
print(df.head()) #Dataset preview
print(df['Recurred'].value_counts())

# Visualization: Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df['Recurred'], palette="prism", hue=df['Recurred'], legend=False)
plt.xticks([0,1], ['No', 'Yes'])
plt.xlabel("Thyroid Reccurence")
plt.ylabel("Count")
plt.title("Class Distribution in Thyroid Cancer Dataset")
plt.show()

#Transform categorical features to numerical
encoder = LabelEncoder()
#Columns of the dataset that require encoding
columns = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function', 'Physical Examination', 'Adenopathy',
            'Pathology', 'Focality', 'Risk', 'T', 'M', 'N', 'Stage', 'Recurred', 'Response']
for col in columns:
     df[col] = encoder.fit_transform(df[col])

#No standardization since that would alter the categorical features too much
X = df.drop(columns=['Recurred']) #Features (input)
y = df['Recurred'] #Target (output)
# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Algorithm: Random Forest Classifier
rf_random_state = 42 #original: 42
rf = RandomForestClassifier(random_state=rf_random_state)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

#Algorithm: Support Vector Machine
svm_random_state = 42 #original: 42
svm = SVC(kernel='linear', random_state=svm_random_state)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

#Algorithm: KNN
n = 5 #original: 5
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

#Algorithm: Logistic Regression
log_random_state = 42
logistic = LogisticRegression(random_state=log_random_state)
logistic.fit(X_train, y_train)
y_pred_log = logistic.predict(X_test)

#results of all the algorithms
results = {
    "Algorithm": ["SVM", "Random Forest", "KNN", "Logistic Regression"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_log)]
}

results_df = pd.DataFrame(results) #turns the dictionary into a DataFrame

# Visualization: Model Accuracy Comparison
plt.figure(figsize=(6, 4))
sns.barplot(x="Algorithm", y="Accuracy", data=results_df, palette="viridis", hue='Accuracy', legend=False)
plt.ylim(0.7, 1)  # Adjusting the range for better comparison
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

print(results_df.head())