import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

path = r"C:\Users\0753780\Downloads\Pathway Selection _ Personality Survey(Sheet1).csv"
df = pd.read_csv(path)

print(df.head())
encoder = LabelEncoder()
df['What pathway are you in?'] = encoder.fit_transform(df['What pathway are you in?'])

y = df['What pathway are you in?']
x = df.drop(columns=['What pathway are you in?', 'Email', 'Name', 'Completion time',
                      'Start time', 'Id'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
log = LogisticRegression(random_state=42)
log.fit(x_train, y_train)
log_pred = log.predict(x_test)

rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

print(f"Accuracy (Logistic Regression): {accuracy_score(y_test, log_pred) * 100:.2f}%")
print(f"Accuracy (Random Forest Classifier): {accuracy_score(y_test, rf_pred) * 100:.2f}%")

cm_log = confusion_matrix(y_test, log_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_log)
disp.plot()
plt.title("Pathway Predictions\nModel: Logistic Regression")
plt.show()

cm_rf = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp.plot()
plt.title("Pathway Predictions\nModel: Random Forest Classifier")
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
plt.ylim(0.4, 0.7)  # Adjusting the range for better comparison
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()