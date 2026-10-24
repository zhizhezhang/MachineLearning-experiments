# -*- coding: utf-8 -*-
"""
机器学习实验8 —— 第2题：打网球问题
根据天气/温度/有风 11 条历史数据训练决策树，预测明天是否打球
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# ---------- 中文显示设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # 历史数据（来自实验报告表格）
    data = pd.DataFrame({
        '天气': ['晴天', '晴天', '阴天', '雨天', '雨天', '雨天',
                '阴天', '晴天', '晴天', '雨天', '雨天'],
        '温度': ['热', '热', '热', '温和', '冷', '冷',
                '冷', '温和', '冷', '温和', '温和'],
        '有风': ['无风', '有风', '无风', '无风', '无风', '有风',
                '有风', '无风', '无风', '无风', '有风'],
        '是否玩': ['否', '否', '是', '是', '是', '否',
                '是', '否', '是', '是', '否'],
    })
    print(f'历史数据:\n{data}\n')

    # 类别特征编码
    encoders = {}
    X_enc = pd.DataFrame()
    for col in ['天气', '温度', '有风']:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(data[col])
        encoders[col] = le
        print(f'{col} 编码映射: {dict(zip(le.classes_, le.transform(le.classes_)))}')

    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(data['是否玩'])  # 否->0, 是->1
    print(f'标签编码映射: {dict(zip(le_y.classes_, le_y.transform(le_y.classes_)))}\n')

    # 训练决策树（小数据集，不限制深度）
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X_enc.values, y_enc)
    print(f'训练集精度: {clf.score(X_enc.values, y_enc):.4f}')

    # 对几种典型的"明天天气"组合进行预测
    tomorrow_cases = pd.DataFrame({
        '天气': ['晴天', '阴天', '雨天', '晴天', '雨天'],
        '温度': ['温和', '热', '冷', '热', '温和'],
        '有风': ['无风', '无风', '有风', '有风', '有风'],
    })
    print('对几种明天天气情况的预测结果：')
    for _, row in tomorrow_cases.iterrows():
        x_new = np.array([
            encoders['天气'].transform([row['天气']])[0],
            encoders['温度'].transform([row['温度']])[0],
            encoders['有风'].transform([row['有风']])[0],
        ]).reshape(1, -1)
        pred = clf.predict(x_new)[0]
        prob = clf.predict_proba(x_new)[0]
        print(f'  天气={row["天气"]}  温度={row["温度"]}  有风={row["有风"]}  '
              f'-> 预测: {"打(是)" if pred == 1 else "不打(否)"}  '
              f'(是概率={prob[1]:.2f}, 否概率={prob[0]:.2f})')

    # 导出决策树结构图
    plt.figure(figsize=(14, 6))
    plot_tree(clf, feature_names=['天气', '温度', '有风'],
              class_names=list(le_y.classes_),
              filled=True, rounded=True, fontsize=9)
    plt.title('打网球问题 —— 决策树结构')
    p = os.path.join(BASE_DIR, 'result3_tennis_tree.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n决策树结构图已保存: {p}')


if __name__ == '__main__':
    main()
