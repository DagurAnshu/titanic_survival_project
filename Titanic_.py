from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

df=pd.read_csv('train_titanic.csv')
#print(df.columns)
print("Missing values per column:")
print(df.isnull().sum())
df_filled=df.fillna(df.mean(numeric_only=True))

df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Deck']=df['Cabin'].str[0]
df['Deck']=df['Deck'].fillna("U")
print(df[['Cabin','Deck']])

print("Missing values per column after filling:")
print(df.isnull().sum())
df = df.drop(['PassengerId'], axis=1)
inputs=df.drop(['Survived'],axis=1)
print(inputs.columns)
output=df[['Survived']]
print(output.columns)

col_types=pd.DataFrame({
    'Column':inputs.columns,
    'Dtype':[inputs[col].dtype for col in inputs.columns],
'Type': ['Categorical' if inputs[col].dtype == 'object' or
                          inputs[col].dtype.name == 'category' else 'Numeric' for col in inputs.columns]
})
print(col_types)

numeric_features=['Fare','Age']
categorical_features=['Sex','Embarked','Pclass']

for col in categorical_features + numeric_features:
    assert col in inputs.columns, f"Column '{col}' not found in inputs"

X_train, X_test, y_train, y_test = train_test_split(inputs,output,test_size=0.1,random_state=42)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

ct=ColumnTransformer(transformers=[('num',MinMaxScaler(),numeric_features),
                                   ('cat',OneHotEncoder(),categorical_features)])
X_train_transformed=ct.fit_transform(X_train)
X_test_transformed=ct.transform(X_test)

models={
    'Logistic Regression':LogisticRegression(),
    'RandomForest':RandomForestClassifier(),
    'SVM':SVC()
}
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

for name,clf in models.items():
    clf.fit(X_train_transformed,y_train)
    preds=clf.predict(X_test_transformed)
    acc=accuracy_score(y_test,preds)
    print(f"Accuracy Score for {name} :  {acc:.4f}")
    print(f"Confusion Matrix for {name}: {confusion_matrix(y_test,preds)}")
    print(f"Classification Report for {name}: {classification_report(y_test,preds)}")
    #print("Accuracy Score for {} is {}",name,acc)
    cm=confusion_matrix(y_test,preds)
    report=classification_report(y_test,preds,output_dict=True)
    report_df=pd.DataFrame(report).transpose().iloc[:-1,:-1]
    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{name} Evaluation", fontsize=16)

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    # Classification Report
    sns.heatmap(report_df, annot=True, cmap='Blues', ax=ax2)
    ax2.set_title("Classification Report")
    ax2.set_xlabel("Metrics")
    ax2.set_ylabel("Classes")
    fig.patch.set_facecolor('#121212')
    ax1.set_facecolor('#1e1e1e')
    ax2.set_facecolor('#1e1e1e')

    for ax in [ax1, ax2]:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('lightgreen')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
print(df.columns)

survival_rates = {}
for name, clf in models.items():
    preds = clf.predict(X_test_transformed)
    survival_rate = np.mean(preds)
    survival_rates[name] = survival_rate


# Streamlit UI
import streamlit as st
st.set_page_config(page_title="Titanic Survival Dashboard", layout="wide")
st.title("🚢 Titanic Survival Prediction Dashboard")

# Sidebar input
st.sidebar.header("🧍 Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
age = st.sidebar.slider("Age", 0, 80, 30)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 50.0)
embarked = st.sidebar.selectbox("Embarked", ['C', 'Q', 'S'])
model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))

# Predict survival
input_df = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'Fare': fare,
    'Embarked': embarked,
    'SibSp': 0,
    'Parch': 0,
    'Deck': 'U'
}])
input_transformed = ct.transform(input_df)
chosen_model = models[model_choice]
prediction = chosen_model.predict(input_transformed)[0]
proba = chosen_model.predict_proba(input_transformed)[0][1]

# Output prediction
st.subheader("🧠 Prediction Result")
st.write("🛟 Survived" if prediction == 1 else "⚓ Did Not Survive")
st.metric(label="Survival Probability", value=f"{proba:.2%}")

# Survival rate comparison
st.subheader("Average Survival Rate per Model")
rate_df = pd.DataFrame(list(survival_rates.items()), columns=['Model', 'Survival Rate'])

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    ax1.bar(rate_df['Model'], rate_df['Survival Rate'], color='skyblue')
    ax1.set_ylabel("Survival Rate")
    ax1.set_ylim(0, 1)
    ax1.set_title("Model-wise Survival Rate")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.pie(rate_df['Survival Rate'], labels=rate_df['Model'], autopct='%1.1f%%', startangle=90)
    ax2.set_title("Survival Rate Distribution")
    st.pyplot(fig2)

# Evaluation metrics
st.subheader("📈 Model Evaluation")
preds_eval = chosen_model.predict(X_test_transformed)
cm = confusion_matrix(y_test, preds_eval)
report = classification_report(y_test, preds_eval, output_dict=True)
report_df = pd.DataFrame(report).transpose().iloc[:-1, :-1]

fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
fig3.patch.set_facecolor('#121212')
ax3.set_facecolor('#1e1e1e')
ax4.set_facecolor('#1e1e1e')

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
ax3.set_title("Confusion Matrix")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

sns.heatmap(report_df, annot=True, cmap='Blues', ax=ax4)
ax4.set_title("Classification Report")
ax4.set_xlabel("Metrics")
ax4.set_ylabel("Classes")

for ax in [ax3, ax4]:
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('lightgreen')

st.pyplot(fig3)

