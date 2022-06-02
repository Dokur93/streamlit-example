import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df=pd.read_csv("cleaned_data.csv")
df.drop("Unnamed: 0",axis=1,inplace=True)
page=st.sidebar.selectbox("Menu",["Main Page","Salary Prediction Page","Classifier Page","Explore Data Page"])
if page=="Salary Prediction Page":
    st.title("Salary Predictions Based On Stack Overflow 2021 Survey")
    st.image("data:image/jpeg;base64,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")
    def category_shortener(category,limit):
        category_map={}
        for i in range(len(category)):
            if category[i]>=limit:
                category_map[category.index[i]]=category.index[i]
            else:
                category_map[category.index[i]]="Other"
        return category_map

    category_map=category_shortener(df["Country"].value_counts(),300)
    df["Country"]=df["Country"].map(category_map)
    country = [
        "Other",
        "United States of America",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "India",
        "Canada",
        "France",
        "Brazil",
        "Spain",
        "Netherlands",
        "Australia",
        "Sweden",
        "Poland",
        "Italy",
        "Russian Federation",
        "Turkey",
        "Norway",
        "Switzerland",
        "Israel",
        "Belgium",
        "Ukraine",
        "Denmark",
        "Finland",
        "Austria"
    ]
    ed_level = [
        "Master's Degree",
        "Bachelor's Degree",
        "Less than a Bachelor's Degree",
        "Doctoral Degree"
    ]
    coun = st.selectbox("Please Choose A Country: ", country)
    ed = st.selectbox("Please Choose Education Level", ed_level)
    experience = st.slider("Please Choose Your Experience Level", 0.0, 50.0, 3.0, 0.5)
    age = st.number_input("Please Enter Your Age")
    gender = st.radio("Please Choose A Gender", ["Man", "Woman"])
    pred_x={"Country":coun,"EdLevel":ed,"YearsCodePro":experience,"Age":age,"Gender":gender,"Salary":0}
    df.loc[len(df)]=pred_x
    df["EdLevel"] = [0 if i == "Less than a Bachelor's Degree" else i for i in df["EdLevel"]]
    df["EdLevel"] = [1 if i == "Bachelor's Degree" else i for i in df["EdLevel"]]
    df["EdLevel"] = [2 if i == "Master's Degree" else i for i in df["EdLevel"]]
    df["EdLevel"] = [3 if i == "Doctoral Degree" else i for i in df["EdLevel"]]
    df = pd.get_dummies(df, columns=["Gender","Country"], drop_first=True)
    ##Models
    df2=df.iloc[:-1]
    avr_sal=df2["Salary"].mean()
    std_sal=df2["Salary"].std()
    y = df2["Salary"].values.reshape(-1, 1)
    x = df2.drop("Salary", axis=1).values
    scaler = StandardScaler()
    y = scaler.fit_transform(y)
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=7)
    model_sec = st.selectbox("Select A ML Algorithm",
                             ["Linear Regression", "Polynomial Regression", "Decision Tree", "Random Forest",
                              "GridSearchCV Random Forest"])
    if model_sec=="Linear Regression":
        lin_reg = LinearRegression()
        model = lin_reg.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        verbose=st.radio("Verbose",["True","False"])
        salary_pred=model.predict(df.iloc[[-1]].drop("Salary",axis=1))*std_sal+avr_sal
        if st.button("Predict Salary"):
            st.balloons()
            st.header(f"Salary Prediction For 2021 is ${salary_pred[0][0]:.02f}")
            df.loc[df["Salary"]==0,"Salary"]=salary_pred[0][0]
            if verbose == "True":
                st.subheader(f"Linear Model For This Data Has;")
                st.write(f"Root Mean Squared Error: {rmse:.02f}", end="\n  ")
                st.write(f"Mean Absolute Error: {mae:.02f}", end="\n  ")
                st.write(f"R^2 Score(R Squared Score): {r2:.02f}")
            st.write(df)
    elif model_sec=="Polynomial Regression":
        degree=st.slider("Degree",1,5,3)
        pol_reg = PolynomialFeatures(degree=degree)
        pol_x = pol_reg.fit_transform(x_train)
        lin_reg2 = LinearRegression()
        model = lin_reg2.fit(pol_x, y_train)
        pol_x_test = pol_reg.transform(x_test)
        y_pred = model.predict(pol_x_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        verbose = st.radio("Verbose", ["True", "False"])
        if verbose == "True":
            st.subheader(f"Polynomial Model For This Data Has;")
            st.write(f"Root Mean Squared Error: {rmse:.02f}", end="\n  ")
            st.write(f"Mean Absolute Error: {mae:.02f}", end="\n  ")
            st.write(f"R^2 Score(R Squared Score): {r2:.02f}")
        transformed_x=pol_reg.transform(df.iloc[[-1]].drop("Salary", axis=1))
        salary_pred = model.predict(transformed_x) * std_sal + avr_sal
        if st.button("Predict Salary"):
            st.balloons()
            st.header(f"Salary Prediction For 2021 is ${salary_pred[0][0]:.02f}")
            df.loc[df["Salary"] == 0, "Salary"] = salary_pred[0][0]
            if verbose == "True":
                st.subheader(f"Polynomial Model For This Data Has;")
                st.write(f"Root Mean Squared Error: {rmse:.02f}", end="\n  ")
                st.write(f"Mean Absolute Error: {mae:.02f}", end="\n  ")
                st.write(f"R^2 Score(R Squared Score): {r2:.02f}")
            st.write(df)
    elif model_sec=="Decision Tree":
        tree = DecisionTreeRegressor(random_state=7)
        model = tree.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        verbose = st.radio("Verbose", ["True", "False"])
        salary_pred = model.predict(df.iloc[[-1]].drop("Salary", axis=1)) * std_sal + avr_sal
        if st.button("Predict Salary"):
            st.balloons()
            st.header(f"Salary Prediction For 2021 is ${salary_pred[0]:.02f}")
            df.loc[df["Salary"] == 0, "Salary"] = salary_pred[0]
            if verbose == "True":
                st.subheader(f"Decision Tree Model For This Data Has;")
                st.write(f"Root Mean Squared Error: {rmse:.02f}", end="\n  ")
                st.write(f"Mean Absolute Error: {mae:.02f}", end="\n  ")
                st.write(f"R^2 Score(R Squared Score): {r2:.02f}")
            st.write(df)
    elif model_sec=="Random Forest":
        forest = RandomForestRegressor(random_state=7)
        model = forest.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        verbose = st.radio("Verbose", ["True", "False"])
        salary_pred = model.predict(df.iloc[[-1]].drop("Salary", axis=1)) * std_sal + avr_sal
        if st.button("Predict Salary"):
            st.balloons()
            st.header(f"Salary Prediction For 2021 is ${salary_pred[0]:.02f}")
            df.loc[df["Salary"] == 0, "Salary"] = salary_pred[0]
            if verbose == "True":
                st.subheader(f"Random Forest Model For This Data Has;")
                st.write(f"Root Mean Squared Error: {rmse:.02f}", end="\n  ")
                st.write(f"Mean Absolute Error: {mae:.02f}", end="\n  ")
                st.write(f"R^2 Score(R Squared Score): {r2:.02f}")
    elif model_sec=="GridSearchCV Random Forest":
        max_depth = [None, 2, 4, 6, 8, 10, 12]
        parameters = {"max_depth": max_depth}
        regressor = RandomForestRegressor(random_state=7)
        gs = GridSearchCV(regressor, parameters, scoring="neg_mean_squared_error")
        gs.fit(x_train, y_train.ravel())
        reg = gs.best_estimator_
        y_pred = reg.predict(x_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        verbose = st.radio("Verbose", ["True", "False"])
        salary_pred = reg.predict(df.iloc[[-1]].drop("Salary", axis=1)) * std_sal + avr_sal
        if st.button("Predict Salary"):
            st.balloons()
            st.header(f"Salary Prediction For 2021 is ${salary_pred[0]:.02f}")
            df.loc[df["Salary"] == 0, "Salary"] = salary_pred[0]
            if verbose == "True":
                st.subheader(f"Cross Validated Random Forest Model For This Data Has;")
                st.write(f"Root Mean Squared Error: {rmse:.02f}", end="\n  ")
                st.write(f"Mean Absolute Error: {mae:.02f}", end="\n  ")
                st.write(f"R^2 Score(R Squared Score): {r2:.02f}")
    else:
        pass

elif page=="Classifier Page":
    st.title("Basic Classification Examples")
    st.header("Exploring Different Classifiers")
    dataset = st.sidebar.selectbox("Select A Dataset", ["Iris", "Breast Cancer", "Wine"])
    classifier = st.sidebar.selectbox("Select A Classifier", ["KNN", "SVM", "Random Forest"])
    if dataset == "Iris":
        df = datasets.load_iris()
    elif dataset == "Breast Cancer":
        df = datasets.load_breast_cancer()
    else:
        df = datasets.load_wine()
    x = df["data"]
    y = df["target"]
    st.write(f"Shape of the dataset: {x.shape}")
    st.write(f"Number of Different Classes: {len(np.unique(y))}")


    def add_parameter(cls_name):
        params = {}
        if cls_name == "KNN":
            k = st.sidebar.slider("K", 1, 15)
            params["K"] = k
        elif cls_name == "SVM":
            c = st.sidebar.slider("C", 0.01, 10.0)
            params["C"] = c
        else:
            max_depth = st.sidebar.slider("Max Depth", 2, 15)
            tree = st.sidebar.slider("Number Of Trees", 1, 100)
            params["Max_Depth"] = max_depth
            params["N_Estimators"] = tree
        return params


    def get_classifier(clf_name, params):
        if clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=params["K"])
        elif clf_name == "SVM":
            clf = SVC(C=params["C"])
        else:
            clf = RandomForestClassifier(n_estimators=params["N_Estimators"], max_depth=params["Max_Depth"],
                                         random_state=7)
        return clf


    params = add_parameter(classifier)
    clf = get_classifier(classifier, params)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=7)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acs = accuracy_score(y_test, y_pred)
    st.write(f"Dataset Name: {dataset}")
    st.write(f"Classifier Name: {classifier}")
    st.write(f"Accuracy:{acs}")
    pca = PCA(2)
    x_projected = pca.fit_transform(x)
    x1 = x_projected[:, 0]
    x2 = x_projected[:, 1]
    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    st.pyplot(fig)

elif page=="Explore Data Page":
    st.title("Exploring The Stack Overflow Survey Data")
    st.write("""
    ### Stack Overflow Developer Survey 2021 Cleaned
    """)
    st.write(df)



    def category_shortener(category, limit):
        category_map = {}
        for i in range(len(category)):
            if category[i] >= limit:
                category_map[category.index[i]] = category.index[i]
            else:
                category_map[category.index[i]] = "Other"
        return category_map
    category_map = category_shortener(df["Country"].value_counts(), 500)
    df["Country"] = df["Country"].map(category_map)
    data = df["Country"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")
    st.write("""#### Number Of Data From Different Countries""")
    st.pyplot(fig1)

    df = pd.read_csv("cleaned_data.csv")
    df.drop("Unnamed: 0", axis=1, inplace=True)
    category_map = category_shortener(df["Country"].value_counts(), 300)
    df["Country"] = df["Country"].map(category_map)
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(x="Country", y="Salary", data=df)
    plt.xticks(rotation=90)
    plt.title("Salary BoxPlots For Every Country")
    st.write("""#### Salary Boxplots For Different Countries""")
    st.pyplot(fig)

    st.write("""#### Salary Distributions""")
    fig=plt.figure(figsize=(12,6))
    sns.histplot(df,x="Salary",kde=True,color="g")
    st.pyplot(fig)

    st.write("""#### Mean Salaries For Different Countries""")
    data=df.groupby("Country")["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write("""#### Mean Salaries For Different Experience Levels""")
    data = df.groupby("YearsCodePro")["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

else:
    st.title("Machine Learning Website With Different Examples")
    st.image("https://www.eurixgroup.com/wp-content/uploads/2021/01/ml-e1610553826718.jpg")
