import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import zipfile
import io
import requests


data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"

response = requests.get(data_url)
with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
    zip_ref.extractall(".")


csv_file_path = "bank-additional/bank-additional-full.csv"


bank_df = pd.read_csv(csv_file_path, sep=';')

#Preprocessing the data
bank_df = pd.get_dummies(bank_df)


X = bank_df.drop('y_yes', axis=1)  
y = bank_df['y_yes']  

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Building the decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

#Evaluating the model
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=["No", "Yes"], fontsize=10)
plt.title("Decision Tree Classifier")
plt.show()
