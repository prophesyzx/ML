import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os

# 传统机器学习模型准确率
models= ['LightGBM','GaussianNB','RandomForest','AdaBoost','SVC','Knn','LDA']
scores = []
# 保存在scores.txt中
with open("model_scores.txt","r",encoding="utf-8") as f:
    scores = f.readline().split("\t")[:-1]
df = pd.DataFrame(columns=['models', 'scores'])
df['models'] = models
df['scores'] = scores
#title
st.markdown("### 初始模型对比")

#defining side bar
st.sidebar.header("Filters:")

#placing filters in the sidebar using unique values.
location = st.sidebar.multiselect(
     "传统机器学习模型：",
     options=df["models"].unique(),
     default=df["models"].unique()
     )


X = np.load("X.npy")
Y = np.load("Y.npy")
# X_train即train_x，训练集的输入，其他类似
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 2024, test_size=0.2,stratify=Y)


# 读取集成学习个体学习器
lgb_y = np.load("lgb_pred.npy")
rfc_y = np.load("rfc_pred.npy")

# 硬投票
def hard_vote(predicted_probas):
    if len(predicted_probas) == 0:
         return None
    from statistics import mode
    pre = []
    for pred in predicted_probas:
        pre.append(pred.argmax(axis=1))
    return [mode(v) for v in np.array(pre).T]

# 软投票
def soft_vote(predicted_probas):
    if len(predicted_probas) == 0:
         return None, None
    sv_predicted_proba = np.mean(predicted_probas, axis=0)
    sv_predicted_proba[:,-1] = 1 - np.sum(sv_predicted_proba[:,:-1], axis=1)
    return sv_predicted_proba, sv_predicted_proba.argmax(axis=1)

# 读取神经网络模型的概率预测结果

#注意这里的路径更改为自己的路径

nnmodels = os.listdir("result")
nn_pred = []
for mod in nnmodels:
        nn_pred.append(np.load("result/" + mod))

# 所有预测值
all_pred = [lgb_y,rfc_y] + nn_pred
#soft_vote_y = soft_vote(all_pred)[1]
#hard_vote_y = hard_vote(all_pred)

#hard_vote_acc = accuracy_score(Y_test,hard_vote_y)
#soft_vote_acc = accuracy_score(Y_test,soft_vote_y)
lgb_acc = accuracy_score(Y_test,lgb_y.argmax(axis=1))
rfc_acc = accuracy_score(Y_test,rfc_y.argmax(axis=1))
# 所有准确率
acc_list = [lgb_acc,rfc_acc] 
for pred in nn_pred:
    acc_list.append(accuracy_score(Y_test,pred.argmax(axis=1)))
#acc_list.append(hard_vote_acc)
#acc_list.append(soft_vote_acc)

# 所有模型名字
model_name = ['LightGBM','RandomForest']
for mod in nnmodels:
        model_name.append(mod.replace(".npy",""))
#model_name.append('Hard Voting')
#model_name.append('Soft Voting')

df2 = pd.DataFrame(columns=["model","accuracy","pred"])
df2['model'] = model_name
df2['accuracy'] = acc_list
df2["pred"] = all_pred
#placing filters in the sidebar using unique values.
compare = st.sidebar.multiselect(
"集成学习个体学习器：",
options=df2['model'].unique(),
default=df2['model'].unique()
)

df_selection = df.query(
     "`models`== @location"
 )

df_selection2 = df2.query(
     "`model`== @compare"
 )
#a dividing line
st.divider()

#defining the chart 
if len(df_selection["models"]) == 0:
     st.markdown("模型不能为空！！！")
else:
    fig_1 = px.bar(df_selection,x="models",y="scores",text="scores",labels={"models":"模型","scores":"准确率"})
    fig_1.update_traces(texttemplate="%{text:.4f}",  # 数字保留2位有效数字
                    textposition="outside")  # 数字显示在外面
    st.plotly_chart(fig_1, use_container_width=True)
st.markdown("### 模型融合对比")
try:
    pred_data = df_selection2["pred"].tolist()
    soft_vote_y = soft_vote(pred_data)[1]
    hard_vote_y = hard_vote(pred_data)
    hard_vote_acc = accuracy_score(Y_test,hard_vote_y)
    soft_vote_acc = accuracy_score(Y_test,soft_vote_y)
    x_name = df_selection2["model"].tolist() + ["Hard Voting","Soft Voting"]
    y_data = df_selection2["accuracy"].tolist() + [hard_vote_acc,soft_vote_acc]
    df2_view = pd.DataFrame(columns=["model","accuracy"])
    df2_view["model"] = x_name
    df2_view["accuracy"] = y_data

    fig_2 = px.bar(df2_view,x="model",y="accuracy",text="accuracy",labels={"model":"模型","accuracy":"准确率"})
    fig_2.update_traces(texttemplate="%{text:.4f}",  # 数字保留2位有效数字
                    textposition="outside")  # 数字显示在外面
    #displaying the chart on the dashboard
    st.plotly_chart(fig_2, use_container_width=True)
    
except:
     st.markdown("模型不能为空！！！")
     