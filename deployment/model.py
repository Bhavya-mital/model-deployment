import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\DELL\Desktop\advertising.csv")
print(df.head())

x = df[['Age','Area Income','Male','Daily Time Spent on Site','Daily Internet Usage']]
y = df['Clicked on Ad']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33,random_state=50)

classifier = LogisticRegression()
classifier.fit(x_train,y_train)

predictions = classifier.predict(x_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(predictions,y_test)
print(score)

# making pickle file of our model
import pickle

pickle.dump(classifier, open("model.pkl", "wb"))