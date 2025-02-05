import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

test_passangerIds = test_df['PassengerId']

#print(train_df.shape)

#Drop Ticket, Fare, and Cabin columns
to_drop = ['PassengerId','Name','Ticket','Fare','Cabin']
train_df.drop(to_drop, inplace=True, axis=1)
test_df.drop(to_drop, inplace=True, axis=1)

#print(train_df.shape)

#Remove Duplicates - No duplicates
#print(train_df[train_df.duplicated(keep = False)])

#Remove Nulls
#print(train_df.isna().sum())
train_df = train_df.dropna(subset=['Age'])
train_df = train_df.dropna(subset=['Embarked'])


#One Hot Encoding using pandas
one_hot_encoded_train_df = pd.get_dummies(train_df, columns = ['Sex','Embarked'], prefix = ['Sex','Embarked'])
one_hot_encoded_test_df = pd.get_dummies(test_df, columns = ['Sex','Embarked'], prefix = ['Sex','Embarked'])

#One Hot Encoding using sklearn
"""
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded_array = encoder.fit_transform(train_df[['Sex','Embarked']])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded_array, columns=encoder.get_feature_names_out(['Sex','Embarked']))
final_df = pd.concat([train_df, one_hot_encoded_df], axis=1).drop(['Sex','Embarked'], axis=1)
"""

X = one_hot_encoded_train_df.drop(columns='Survived')
y = one_hot_encoded_train_df['Survived']
X_test = one_hot_encoded_test_df

model = DecisionTreeClassifier()
model.fit(X, y)

y_pred = model.predict(X_test)
prediction_df = pd.DataFrame({'PassengerId': test_passangerIds, 'Survived': y_pred})
print(prediction_df)

prediction_df.to_csv('output.csv', index=False)
