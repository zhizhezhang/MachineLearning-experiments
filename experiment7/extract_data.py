# -*- coding: utf-8 -*-
"""从实验报告 docx 中提取糖尿病数据集表格，保存为 diabetes.csv"""
from docx import Document
import csv

doc = Document(r'D:\Documents\1课程资料\1机器学习\机器学习实验7-SVM.docx')

header = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

rows = []
for table in doc.tables:
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        # 跳过表头行（含中文或非数字）
        try:
            vals = [float(x) for x in cells]
        except ValueError:
            continue
        rows.append(vals)

print(f'共提取 {len(rows)} 条数据')

with open(r'/\experiment7\diabetes.csv',
          'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print('已保存 diabetes.csv')
