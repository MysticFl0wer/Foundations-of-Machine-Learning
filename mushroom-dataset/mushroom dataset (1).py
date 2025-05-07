import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from csv import DictReader
#Import models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def clean_data(dataFrame, cols=[]):
    encoder = LabelEncoder()
    for column in cols:
        dataFrame[column] = encoder.fit_transform(dataFrame[column])
    
    return dataFrame

categorial_features = ['poisonous', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                       'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                       'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
                       'ring-number', 'ring-type','spore-print-color' ,'population', 'habitat']

path = r"agaricus-lepiota.txt"
file = open(path)
#read the data as a csv to convert to DataFrame
#this is done because the original file doesn't have the column names
csvFile = DictReader(file, fieldnames=categorial_features, delimiter=',')
df = pd.DataFrame(csvFile) #Turn csv to DataFrame
file.close() #close the file to save resources
#preview of dataset
print(df.head())
# Visualization: Class Distribution
print(df['poisonous'].value_counts())
plt.figure(figsize=(6,4))
sns.countplot(x=df['poisonous'], palette="prism", hue=df['poisonous'], legend=False)
plt.xticks([0,1], ['Poisonous', 'Edible'])
plt.xlabel("Mushroom Edibility")
plt.ylabel("Count")
plt.title("Class Distribution in Mushroom Dataset")
plt.show()

#Handle missing values and encode categorical features
df = df.drop(df[(df["stalk-root"] == "?")].index, axis=0) #Remove rows that have missing values
le = LabelEncoder()
#encode features
for i in categorial_features:
    df[i] = le.fit_transform(df[i])

x = df.drop(columns=['poisonous'])
y = df['poisonous']
#Create training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#Evaluation Metric
#Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
#Logistic Regression
log = LogisticRegression(random_state=42)
log.fit(x_train, y_train)
log_pred = log.predict(x_test)

#Algorithm: Support Vector Machine
svm_random_state = 42 #original: 42
svm = SVC(kernel='linear', random_state=svm_random_state)
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)

results = {
    "Algorithm": ['Random Forest Classification', 'Logistic Regression', "SVM"],
    "Accuracy": [accuracy_score(y_test, y_pred), accuracy_score(y_test, log_pred), accuracy_score(y_test, y_pred_svm)]
}
#Compare Algorithms
results_df = pd.DataFrame(results)
print(results_df.head())
# Visualization: Model Accuracy Comparison
plt.figure(figsize=(6, 4))
sns.barplot(x="Algorithm", y="Accuracy", data=results_df, palette="viridis", hue='Accuracy', legend=False)
plt.ylim(0.7, 1)  # Adjusting the range for better comparison
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()