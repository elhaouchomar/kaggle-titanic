import pandas as pd
import numpy as np

def process_data(train_df, test_df):
    
    train_df['IsTrain'] = 1
    test_df['IsTrain'] = 0
    test_df['Survived'] = np.nan
    
    data = pd.concat([train_df, test_df], ignore_index=True)
    
    data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Family_Survival'] = 0.5
    
    for _, grp_df in data.groupby(['Last_Name', 'Fare']):
        if len(grp_df) != 1:
            for ind, row in grp_df.iterrows():
                others = grp_df.drop(ind)['Survived']
                if others.max() == 1.0:
                    data.loc[data['PassengerId'] == row['PassengerId'], 'Family_Survival'] = 1
                elif others.min() == 0.0:
                    data.loc[data['PassengerId'] == row['PassengerId'], 'Family_Survival'] = 0
                    
    for _, grp_df in data.groupby('Ticket'):
        if len(grp_df) != 1:
            for ind, row in grp_df.iterrows():
                if (row['Family_Survival'] == 0) or (row['Family_Survival'] == 0.5):
                    others = grp_df.drop(ind)['Survived']
                    if others.max() == 1.0:
                        data.loc[data['PassengerId'] == row['PassengerId'], 'Family_Survival'] = 1
                    elif others.min() == 0.0:
                        data.loc[data['PassengerId'] == row['PassengerId'], 'Family_Survival'] = 0

    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                           'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
    data['Title'] = data['Title'].map({"Mr": 1,
                                       "Miss": 2,
                                       "Mrs": 3,
                                       "Master": 4,
                                       "Rare": 5}).fillna(0).astype(int)

    data['Sex'] = data['Sex'].map({
        'female': 1,
        'male': 0}).astype(int)

    data['Age'] = data.groupby(['Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))
    data['Age'] = data['Age'].fillna(data['Age'].median())
    
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age'] = 4
    data['Age'] = data['Age'].astype(int)

    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)

    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Last_Name']
    data = data.drop(columns=drop_elements)
    
    train_processed = data[data['IsTrain'] == 1].drop(columns=['IsTrain'])
    test_processed = data[data['IsTrain'] == 0].drop(columns=['IsTrain', 'Survived'])
    
    return train_processed, test_processed