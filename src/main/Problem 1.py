import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# load data
df = pd.read_csv('src/main/resources/train.csv')

# preprocess data
category_col = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 'Internet_Access',
                'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Gender']
df = pd.get_dummies(df, columns=category_col, drop_first=True)

remaining_category_col = ['Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Distance_from_Home']
df = pd.get_dummies(df, columns=remaining_category_col, drop_first=True)

# convert exam score to binary class
df['Exam_Score_Binary'] = df['Exam_Score'].apply(lambda x: 1 if x >= 75 else 0)
X = df.drop(['Exam_Score', 'Exam_Score_Binary'], axis=1).values
y = df['Exam_Score_Binary'].values

# normalizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

