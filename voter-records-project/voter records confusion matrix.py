import pandas as pd
from csv import DictReader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
import seaborn as sns


def handle_missing_values(dataFrame, cols=[]):
    for column in cols: #for each column
        dataFrame = dataFrame.drop(dataFrame[(dataFrame[column] == "?")].index, axis=0) #Remove rows that have missing values
    
    return dataFrame


def clean_data(dataFrame, cols=[]):
    dataFrame = handle_missing_values(dataFrame, cols)
    encoder = LabelEncoder()
    for column in cols:
        dataFrame[column] = encoder.fit_transform(dataFrame[column])
    
    return dataFrame

dataset_path = r"voter-records.txt"
file = open(dataset_path)

features = ['political-party', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
            'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
              'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending',
              'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
csvFile = DictReader(file, fieldnames=features, delimiter=',')
df = pd.DataFrame(csvFile) #Turn csv to DataFrame
file.close() #close the file to save resources
df = clean_data(df, features)
print(df.head())
print("====================================")

Y = df['political-party']
X = df.drop(columns=['political-party'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
print(f"Random Forest Classifier Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%")
#Cross Validation
k_folds = KFold(n_splits = 5)
scores = cross_val_score(rf, X, Y, cv = k_folds)
print(f"Average Cross Validation Score (Random Forest): {scores.mean():.2f}")
print("====================================")

cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Voter Records Predictions\nModel: Random Forest Classifier\n 0 = Democrat  1 = Republican")
plt.show()

log = LogisticRegression(random_state=42)
log.fit(x_train, y_train)
log_pred = log.predict(x_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_pred) * 100:.2f}%")
#Cross Validation
k_folds = KFold(n_splits = 5)
scores = cross_val_score(log, X, Y, cv = k_folds)
print(f"Average Cross Validation Score (Logistic Regression): {scores.mean():.2f}")
print("====================================")

cm_log = confusion_matrix(y_test, log_pred)
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log)
disp_log.plot()
plt.title("Voter Records Predictions\nModel: Logistic Regression\n 0 = Democrat  1 = Republican")
plt.show()

results = {
    "Algorithm": ['Random Forest Classification', 'Logistic Regression'],
    "Accuracy": [accuracy_score(y_test, rf_pred), accuracy_score(y_test, log_pred)]
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