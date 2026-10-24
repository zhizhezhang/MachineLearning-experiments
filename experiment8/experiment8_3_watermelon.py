# -*- coding: utf-8 -*-
"""
机器学习实验8 —— 第3题：西瓜好坏判断问题
根据色泽/根蒂/敲声/纹理/脐部/触感 17 条数据训练决策树，判断西瓜好坏
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------- 中文显示设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # 17 条历史数据（来自实验报告表格）
    data = pd.DataFrame({
        '色泽': ['青绿', '乌黑', '乌黑', '青绿', '浅白', '青绿', '乌黑', '乌黑',
                '乌黑', '青绿', '浅白', '浅白', '青绿', '浅白', '乌黑', '浅白', '青绿'],
        '根蒂': ['蜷缩', '蜷缩', '蜷缩', '蜷缩', '蜷缩', '稍蜷', '稍蜷', '稍蜷',
                '稍蜷', '硬挺', '硬挺', '蜷缩', '稍蜷', '稍蜷', '稍蜷', '蜷缩', '蜷缩'],
        '敲声': ['浊响', '沉闷', '浊响', '沉闷', '浊响', '浊响', '浊响', '浊响',
                '沉闷', '清脆', '清脆', '浊响', '浊响', '沉闷', '浊响', '浊响', '沉闷'],
        '纹理': ['清晰', '清晰', '清晰', '清晰', '清晰', '清晰', '稍糊', '清晰',
                '稍糊', '清晰', '模糊', '模糊', '稍糊', '稍糊', '清晰', '模糊', '稍糊'],
        '脐部': ['凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '稍凹', '稍凹', '稍凹',
                '稍凹', '平坦', '平坦', '平坦', '凹陷', '凹陷', '稍凹', '平坦', '稍凹'],
        '触感': ['硬滑', '硬滑', '硬滑', '硬滑', '硬滑', '软粘', '软粘', '硬滑',
                '硬滑', '软粘', '硬滑', '软粘', '硬滑', '硬滑', '软粘', '硬滑', '硬滑'],
        '好坏': ['好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜',
                '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜'],
    })
    print(f'数据集大小: {data.shape}')
    print(f'好瓜数量: {(data["好坏"] == "好瓜").sum()}，'
          f'坏瓜数量: {(data["好坏"] == "坏瓜").sum()}\n')

    # 按实验报告给出的编码映射手工编码
    # 色泽：0青绿 1乌黑 2浅白；根蒂：0蜷缩 1稍蜷 2硬挺
    # 敲声：0清脆 1浊响 2沉闷；脐部：0凹陷 1稍凹 2平坦
    # 纹理：0稍糊 1清晰 2模糊；触感：0硬滑 1软粘
    encode_map = {
        '色泽': {'青绿': 0, '乌黑': 1, '浅白': 2},
        '根蒂': {'蜷缩': 0, '稍蜷': 1, '硬挺': 2},
        '敲声': {'清脆': 0, '浊响': 1, '沉闷': 2},
        '纹理': {'稍糊': 0, '清晰': 1, '模糊': 2},
        '脐部': {'凹陷': 0, '稍凹': 1, '平坦': 2},
        '触感': {'硬滑': 0, '软粘': 1},
    }
    X = data[list(encode_map.keys())].replace(encode_map).values
    y = (data['好坏'] == '好瓜').astype(int).values  # 好瓜->1, 坏瓜->0
    print('编码映射（按实验报告）：')
    for k, v in encode_map.items():
        print(f'  {k}: {v}')

    # 训练决策树（信息增益 = entropy）
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X, y)
    print(f'\n训练集精度: {clf.score(X, y):.4f}')
    print(f'各特征重要性:')
    for name, imp in zip(encode_map.keys(), clf.feature_importances_):
        print(f'  {name}: {imp:.4f}')

    # 预测两个新西瓜
    # 样本A：青绿 / 蜷缩 / 浊响 / 清晰 / 凹陷 / 硬滑  —— 典型好瓜
    # 样本B：乌黑 / 稍蜷 / 沉闷 / 稍糊 / 平坦 / 软粘  —— 典型坏瓜
    new_melons = pd.DataFrame({
        '色泽': ['青绿', '乌黑'],
        '根蒂': ['蜷缩', '稍蜷'],
        '敲声': ['浊响', '沉闷'],
        '纹理': ['清晰', '稍糊'],
        '脐部': ['凹陷', '平坦'],
        '触感': ['硬滑', '软粘'],
    })
    print('\n新西瓜预测结果：')
    for i, row in new_melons.iterrows():
        x_new = np.array([encode_map[c][row[c]] for c in encode_map]).reshape(1, -1)
        pred = clf.predict(x_new)[0]
        prob = clf.predict_proba(x_new)[0]
        print(f'  样本{i + 1}: {dict(row)}')
        print(f'    -> 预测: {"好瓜" if pred == 1 else "坏瓜"}  '
              f'(好瓜概率={prob[1]:.2f}, 坏瓜概率={prob[0]:.2f})')

    # 导出决策树结构图
    plt.figure(figsize=(14, 7))
    plot_tree(clf, feature_names=list(encode_map.keys()),
              class_names=['坏瓜', '好瓜'],
              filled=True, rounded=True, fontsize=9)
    plt.title('西瓜好坏判断 —— 决策树结构')
    p = os.path.join(BASE_DIR, 'result4_watermelon_tree.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n决策树结构图已保存: {p}')


if __name__ == '__main__':
    main()
