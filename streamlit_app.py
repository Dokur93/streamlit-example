import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
titanic=sns.load_dataset("titanic")
titanic=titanic[["survived","pclass","sex","age","sibsp","parch","fare","embarked"]]
st.sidebar.image("https://pngimg.com/uploads/titanic/titanic_PNG9.png")
pclass=st.sidebar.number_input("Bilet Sınıfınız: ",min_value=1,step=1,max_value=3)
sex=st.sidebar.radio("Cinsiyetiniz: ", titanic["sex"].unique())
age=st.sidebar.number_input("Yaşınız: ",min_value=0,step=1)
sibsp=st.sidebar.number_input("Yakın Sayınız: ",min_value=0,step=1)
parch=st.sidebar.number_input("Çocuk Sayınız: ",min_value=0,step=1)
fare=st.sidebar.number_input("Bilet Fiyatınız: ",min_value=0)
embarked=st.sidebar.selectbox("Bindiğiniz Liman: ", ["C","S","Q"])
if sex=="male":
    sex_male=1
elif sex=="female":
    sex_male=0
if embarked=="C":
    embarked_Q=0
    embarked_S=0
elif embarked=="Q":
    embarked_Q = 1
    embarked_S = 0
elif embarked=="S":
    embarked_Q = 0
    embarked_S = 1
titanic=pd.get_dummies(titanic,columns=["sex","embarked"],drop_first=True)
ermean=np.ceil(titanic[titanic["sex_male"]==1]["age"].mean())
kmean=np.ceil(titanic[titanic["sex_male"]==0]["age"].mean())
titanic.loc[titanic["sex_male"]==1,"age"]=titanic.loc[titanic["sex_male"]==1,"age"].fillna(ermean)
titanic.loc[titanic["sex_male"]==0,"age"]=titanic.loc[titanic["sex_male"]==0,"age"].fillna(kmean)
y=titanic["survived"]
x=titanic.drop("survived",axis=1)
tree=DecisionTreeClassifier()
model=tree.fit(x,y)
skor=np.round(model.score(x,y)*100,2)
tahmin=[pclass,sex_male,age,sibsp,parch,fare,embarked_Q,embarked_S]
sans=model.predict([tahmin])[0]
if st.sidebar.button("Hesapla"):
    if sans==1:
        st.subheader(f"{skor} ihtimalle Yaşardınız!")
        st.balloons()
    elif sans==0:
        st.subheader(f"{skor} ihtimalle Ölürdünüz!")
        st.image("https://www.belfasttelegraph.co.uk/news/northern-ireland/15fcc/40423573.ece/AUTOCROP/w1240h700/-09_new_24480000_I2.jpg")
