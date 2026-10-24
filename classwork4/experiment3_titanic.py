import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    'PassengerId': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    'Pclass': [1,1,1,3,3,1,2,1,1,1,1,1,2,3,1,1,3,1,1,1],
    'Sex': [1,1,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0],
    'Age': [38,35,54,16,4,58,34,28,19,49,65,45,29,25,23,46,40,23,21,47],
    'SibSp': [1,1,0,0,1,0,0,0,3,1,0,1,0,0,3,1,1,0,0,0],
    'Parch': [0,0,0,0,1,0,0,0,2,0,1,0,0,0,2,0,1,1,1,0],
    'Fare': [71.2,53.1,51.9,79.2,16.7,26.6,13.0,35.5,263.0,76.7,62.0,83.4,10.5,7.65,263.0,61.2,15.5,63.4,77.3,52.0],
    'Embarked': [1,3,3,1,3,3,3,3,3,1,1,3,3,3,3,3,2,1,3,3],
    'Survived': [1,1,0,1,1,1,1,1,0,1,0,0,1,0,1,0,0,1,0,0]
}

df = pd.DataFrame(data)
X = df.drop(columns=['PassengerId', 'Survived'])
y = df['Survived']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print('回归系数：')
for col, coef in zip(X.columns, model.coef_[0]):
    print(f'{col}: {coef:.4f}')
print(f'截距：{model.intercept_[0]:.4f}')

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

print('\n预测结果：')
result = pd.DataFrame({
    'PassengerId': df['PassengerId'],
    '真实Survived': y,
    '预测Survived': y_pred,
    '幸存概率': np.round(y_prob, 4)
})
print(result.to_string(index=False))

print(f'\n准确率：{accuracy_score(y, y_pred):.4f}')
print('混淆矩阵：')
print(confusion_matrix(y, y_pred))
print('分类报告：')
print(classification_report(y, y_pred))