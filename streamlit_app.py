
import streamlit as st

import seaborn as sns

import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

import graphviz

st.sidebar.image("https://pngimg.com/uploads/titanic/titanic_PNG9.png")

titanic=sns.load_dataset('titanic')

titanic=titanic.dropna()

titanic=titanic[["survived","pclass","sex","age","sibsp","parch","embarked"]]

pclass=st.sidebar.number_input("pclass giriniz",step=1)

sex=st.sidebar.selectbox("Cinsiyet",["male","female"])

age=st.sidebar.number_input("Yaş",step=1)

sibsp=st.sidebar.number_input("Gemideki Yakın Sayınız",step=1)

parch=st.sidebar.number_input("Gemideki çocuklarınızın sayısı",step=1)

embarked=st.sidebar.selectbox("Liman",["C","S","Q"])

survived=0

tahmin={"survived":survived,"pclass":pclass,"sex":sex,"age":age,"sibsp":sibsp,"parch":parch,"embarked":embarked}

titanic=titanic.append(tahmin,ignore_index=True)

titanic=pd.get_dummies(titanic,columns=["sex","embarked"],drop_first=True)

y=titanic[["survived"]]

x=titanic.drop("survived",axis=1)

tree=DecisionTreeClassifier()

model=tree.fit(x,y)

st.write("Skorunuz",model.score(x,y))

sonuc=model.predict(x.iloc[[-1]])[0]

if sonuc==1:

    st.balloons()

    st.title("Yaşadınız")

else:

    st.title("Öldünüz")
