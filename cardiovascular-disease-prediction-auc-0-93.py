#!/usr/bin/env python
# coding: utf-8

# <h3 
#     align="center" 
#     style="font-size: 35px; padding:20px; font-family:Georgia; text-align:center; display:block; border-radius:20px;background-color:#454545">
#     <span style="color: white;">  🩺 Risk Factors And Prediction 🩺 </span>
# </h3>

# ![1_CQXQxHDKi0Q2IpdjhufEcw.jpg](attachment:eb42c45d-6284-43fd-adb5-f26feffb83ca.jpg)

# <h3 
#     align="center" 
#     style="font-size: 35px; padding:20px; font-family:Georgia; text-align:center; display:block; border-radius:20px;background-color:#454545">
#     <span style="color: white;">📚 Importing Libraries 📚 </span>
# </h3>

# In[1]:


import pandas as pd
import numpy as np

# Visualization
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go
import matplotlib.pyplot as plt
colors = px.colors.sequential.Plasma_r

# Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc

# Warnings
import warnings 
warnings. filterwarnings('ignore')


# <h3 
#     align="center" 
#     style="font-size: 35px; padding:20px; font-family:Georgia; text-align:center; display:block; border-radius:20px;background-color:#454545">
#     <span style="color: white;">⏳ Loading the dataset ⏳</span>
# </h3>

# In[2]:


heart = pd.read_csv('CVD_cleaned.csv')


# In[3]:


heart.head()


# <h3 
#     align="center" 
#     style="font-size: 35px; padding:20px; font-family:Georgia; text-align:center; display:block; border-radius:20px;background-color:#454545">
#     <span style="color: white;">🧠 Understanding of data 🧠</span>
# </h3>

# In[4]:


heart.info()


# In[5]:


heart.shape


# <h3 
#     align="center" 
#     style="font-size: 35px; padding:20px; font-family:Georgia; text-align:center; display:block; border-radius:20px;background-color:#454545">
#     <span style="color: white;">🧹 Data Cleaning 🧹</span>
# </h3>

# In[6]:


# Checking if there are any null values in the dataset or not
heart.isnull().sum()


# In[7]:


# Convering the column names into lower case and replacing the space with an underscore
heart.columns = heart.columns.str.lower().str.replace(" ", "_")

#Changing the name of a big column

heart.rename(columns = {'height_(cm)' : 'height', 'weight_(kg)' : 'weight', 'green_vegetables_consumption' : 'vegetables_consumption', 'friedpotato_consumption' : 'potato_consumption'}, inplace = True)


# In[8]:


# With the help of for loop, we will now check if there are any typos in the categorical columns or not
for col in heart.select_dtypes(include = "object"):
    print(f"Column name: {col}")
    print(heart[col].unique())
    print('\n', '-'*80, '\n')


# In[9]:


heart['checkup'] = heart['checkup'].replace('Within the past 2 years', 'Past 2 years')
heart['checkup'] = heart['checkup'].replace('Within the past year', 'Past 1 year')
heart['checkup'] = heart['checkup'].replace('Within the past 5 years', 'Past 5 years')
heart['checkup'] = heart['checkup'].replace('5 or more years ago', 'More than 5 years')


heart['diabetes'] = heart['diabetes'].replace('No, pre-diabetes or borderline diabetes', 'No Pre Diabetes')
heart['diabetes'] = heart['diabetes'].replace('Yes, but female told only during pregnancy', 'Only during pregnancy')

heart['age_category'] = heart['age_category'].replace('18-24', 'Young')
heart['age_category'] = heart['age_category'].replace('25-29', 'Adult')
heart['age_category'] = heart['age_category'].replace('30-34', 'Adult')
heart['age_category'] = heart['age_category'].replace('35-39', 'Adult')
heart['age_category'] = heart['age_category'].replace('40-44', 'Mid-Aged')
heart['age_category'] = heart['age_category'].replace('45-49', 'Mid-Aged')
heart['age_category'] = heart['age_category'].replace('50-54', 'Mid-Aged')
heart['age_category'] = heart['age_category'].replace('55-59', 'Senior-Adult')
heart['age_category'] = heart['age_category'].replace('60-64', 'Senior-Adult')
heart['age_category'] = heart['age_category'].replace('65-69', 'Elderly')
heart['age_category'] = heart['age_category'].replace('70-74', 'Elderly')
heart['age_category'] = heart['age_category'].replace('75-79', 'Elderly')
heart['age_category'] = heart['age_category'].replace('80+', 'Elderly')


# In[10]:


col = ['alcohol_consumption', 'fruit_consumption', 'vegetables_consumption', 'potato_consumption']

for i in col:
    heart[i] = heart[i].astype(int)


# <div style="border-radius:10px; border:#000000 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=black>📊 Insights:</font></h3>
# 
# * The are **0 null values** in the dataset
# * Some **column names and their values** were changed
# * **Data types** of a few columns were changed

# <h3 
#     align="center" 
#     style="font-size: 35px; padding:20px; font-family:Georgia; text-align:center; display:block; border-radius:20px;background-color:#454545">
#     <span style="color: white;">🔎 Feature Engineering 🔎</span>
# </h3>

# In[11]:


# Define BMI ranges and labels for each group
bmi_bins = [12.02, 18.3, 26.85, 31.58, 37.8, 100]
bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese I', 'Obese II']
heart['bmi_group'] = pd.cut(heart['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)


# In[12]:


column_to_move = heart.pop('bmi_group')
heart.insert(14, 'bmi_group', column_to_move)


# In[13]:


heart['bmi_group'] = heart['bmi_group'].astype('object')


# <div style="border-radius:10px; border:#000000 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=black>📊 Insights:</font></h3>
# 
# * New feature of **bmi_group** was created that consisted of bmi bins
# * The **position** was changed and it's **data type** as well

# <h3 
#     align="center" 
#     style="font-size: 35px; padding:20px; font-family:Georgia; text-align:center; display:block; border-radius:20px;background-color:#454545">
#     <span style="color: white;">📊 Exploratory Data Analysis 📊</span>
# </h3>

# ---
# ### Descriptive Analysis
# ---

# In[14]:


heart.describe(include = 'O')


# In[15]:


heart.describe().T


# <div style="border-radius:10px; border:#000000 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=black>📊 Insights:</font></h3>
# 
# * The analysis revealed that people in the dataset are mostly in a **good health with a normal weight** and the **last checkup was 1 year ago**. They also do **exercise**
# * Furthermore, the **average heigh of poeple is 170cm** with an **average weight of 83Kg and bmi of 28.62** as well

# ---
# ### Identification of Risk Factors
# ---

# #### 1. Health Assessment

# In[16]:


fig1 = px.histogram(heart, x="general_health", color = 'general_health', color_discrete_sequence = colors, title="1. Distribution of General Health")
fig1.update_layout(plot_bgcolor='white')
fig1.show()
print('\n', "="*80, '\n')

fig2 = px.histogram(heart, x="general_health", color = 'heart_disease', color_discrete_sequence = colors, barmode = 'group', title="2. General Health with respect to Heart Disease")
fig2.update_layout(plot_bgcolor='white')
fig2.show()
print('\n', "="*80, '\n')


# #### 2. Demographic Analysis

# In[17]:


sex_counts = heart['sex'].value_counts()
age_category_counts = heart['age_category'].value_counts()

fig1 = px.bar(x=sex_counts.index, y=sex_counts.values, color=sex_counts.index, color_discrete_sequence = colors, labels={'x': 'Sex', 'y': 'Count'})
fig1.update_layout(title="1. Distribution of gender in the Dataset", xaxis_title="", yaxis_title="Count", plot_bgcolor='white')
fig1.show()
print('\n', "="*80, '\n')

fig2 = px.histogram(heart, x="sex", color='heart_disease', barmode='group', color_discrete_sequence= colors, title="2. Checking which gender is more susceptible to Heart Disease?")
fig2.update_layout(xaxis_title="Gender", yaxis_title="Count", legend_title="Heart Disease", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
fig2.show()
print('\n', "="*80, '\n')

grouped_data = heart.groupby(['sex', 'heart_disease'], as_index=False)['bmi'].median()
fig = px.bar(grouped_data, x='sex', y='bmi', color='heart_disease', color_discrete_sequence = colors, barmode = 'group', title="3. Checking  gender and their average bmi based on heart disease?")
fig.update_layout(xaxis_title="Gender", yaxis_title="Average BMI", legend_title="Heart Disease", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
fig.show()


# In[18]:


fig1 = px.bar(x=age_category_counts.index, y=age_category_counts.values, color=age_category_counts.index, color_discrete_sequence = colors ,labels={'x': 'Age Category', 'y': 'Count'})
fig1.update_layout(title="1. Distribution of Age Categories in the Dataset", xaxis_title="", yaxis_title="Count", plot_bgcolor='white')
fig1.show()
print('\n', "="*80, '\n')

fig2 = px.histogram(heart, x="age_category", color='heart_disease', barmode='group', color_discrete_sequence= colors, title="2. Checking which age group is more susceptible to Heart Disease?")
fig2.update_layout(xaxis_title="age_category", yaxis_title="Count", legend_title="Heart Disease", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
fig2.show()
print('\n', "="*80, '\n')

grouped_data = heart.groupby(['age_category', 'heart_disease'], as_index=False)['bmi'].median()
fig3 = px.bar(grouped_data, x='age_category', y='bmi', color='heart_disease', color_discrete_sequence = colors, barmode = 'group', title="3. Checking  age groups and their average bmi based on heart disease?")
fig3.update_layout(xaxis_title="Age Group", yaxis_title="Average BMI", legend_title="Heart Disease", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
fig3.show()


# #### 3. Impact of Lifestyle Analysis

# In[19]:


def create_bar_chart(data, x_col, y_col, color_col, title, x_label, y_label):
    grouped_data = data.groupby([x_col, color_col]).size().reset_index(name='count')
    fig = px.bar(grouped_data, x=x_col, y=y_col, color=color_col, color_discrete_sequence=colors, title=title, labels={x_col: x_label, y_col: y_label}, barmode='group', category_orders={x_col: ["No", "Yes"], color_col: ["No", "Yes"]} )
    fig.update_layout(plot_bgcolor='white')
    fig.show()
    print('\n', "=" * 80, '\n')
    
create_bar_chart(heart, 'exercise', 'count', 'heart_disease', '1. Impact of Exercise on Heart Disease', 'Exercise', 'Count')
create_bar_chart(heart, 'smoking_history', 'count', 'heart_disease', '2. Impact of Smoking on Heart Disease', 'Smoking History', 'Count')

columns = ['alcohol_consumption', 'fruit_consumption', 'vegetables_consumption', 'potato_consumption']
titles = ["Alcohol Consumption", "Fruit Consumption", "Vegetables Consumption", "Potato Consumption"]

for i, col in enumerate(columns):
    grouped_data = heart.groupby(['age_category', 'heart_disease'], as_index=False)[col].median()  # Use median here
    fig = px.bar(grouped_data, x='age_category', y=col, color='heart_disease', color_discrete_sequence=colors, barmode='group', title=f"{i + 4}. Impact of {titles[i]} on Heart Disease")
    fig.update_layout(xaxis_title="Age Group", yaxis_title=f"Median {titles[i]}", legend_title="Heart Disease", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
    fig.show()
    print('\n', "="*80, '\n')


# <div style="border-radius:10px; border:#000000 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=black>📊 Insights:</font></h3>
# 
# * The risk factor analysis revealed that people who have a **poor health have more chances of heart disease**. 
#     
# * In terms of **gender**, the data shows a **high ratio of male diagnosed with heart disease** and they have a **bit hight bmi as compared to females**
#     
# * According to age groups, the **elderly poeple have the highest count** for heart disease. However, **mid aged people having heart disease tend to have a higher bmi**
#     
# * In terms of **exercise**, it doesn't play a significant role. However, **smoking is an important factor for heart disease poeple**.
#     
# * The **alcohol consumption** has a significant impact on heart disease, however, it doesn't affect much when a person is in young age. Furthermore, the analysis revaled that, **fruit consumption isn't having much effect on poeple with heart disease**.
#     
#     
# * Similarly, **green vegetables consumption** has a significant impact on heart disease, specially, it affects more when a person is in adult age. Furthermore, the analysis revaled that, **fried potato consumption isn't having any effect on poeple with heart disease**.

# ---
# ### Correlation Analysis
# ---

# In[20]:


fig1 = px.histogram(heart, x="depression", color="heart_disease", barmode = 'group', color_discrete_sequence=colors, title="1. Correlation between Depression and Heart Disease")
fig1.update_layout(plot_bgcolor='white')
fig1.show()

fig2 = px.histogram(heart, x="diabetes", color="heart_disease", barmode = 'group', color_discrete_sequence=colors, title="2. Correlation between Diabetes and Heart Disease")
fig2.update_layout(plot_bgcolor='white')
fig2.show()

fig3 = px.box(heart, x="diabetes", y="bmi", title="3. BMI levels of poeple dealing with diabetes and heart disease", color="heart_disease", color_discrete_sequence=colors)
fig3.update_layout(xaxis_title= "Diabetes Status", yaxis_title= "BMI", xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor='white')
fig3.show()


# <div style="border-radius:10px; border:#000000 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=black>📊 Insights:</font></h3>
# 
# * The correlation analysis revealed that people who have a **depression have more chances of heart disease**. 
#     
# * Similarly, **diabetes has a significant impact** on heart disease, specially, people with diabetes and heart disease **have a higher bmi on average**.

# <h3 
#     align="center" 
#     style="font-size: 35px; padding:20px; font-family:Georgia; text-align:center; display:block; border-radius:20px;background-color:#454545">
#     <span style="color: white;">⚙️ Data Preprocessing ⚙️</span>
# </h3>

# ---
# ### 1. One-Hot Encoding
# ---

# In[21]:


heart['heart_disease'] = heart['heart_disease'].map({'Yes':1, 'No':0})
cat=['sex', 'smoking_history']

OH_Encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH = OH_Encoder.fit_transform(heart[cat])
cols = OH_Encoder.get_feature_names_out(cat)
OH = pd.DataFrame(OH, columns=cols)
heart = heart.drop(cat,axis=1)
heart = pd.concat([heart, OH], axis =1) 


# ---
# ### 2. Label Encoding
# ---

# In[22]:


categorical_columns = ['general_health', 'checkup', 'exercise', 'skin_cancer', 'other_cancer', 'depression', 'diabetes', 'arthritis', 'age_category', 'bmi_group']

# Initialize LabelEncoder

label_encoder = LabelEncoder()

# Apply label encoding to each ordinal categorical column

for col in categorical_columns:
    heart[col] = label_encoder.fit_transform(heart[col])


# ---
# ### 3. Class Imbalance
# ---

# In[23]:


# Checking the class Imbalance

heart['heart_disease'].value_counts()


# In[24]:


X = heart.drop("heart_disease", axis = 1)
y = heart['heart_disease']


# In[25]:


smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)


# ---
# ### 4. Splitting into training and testing
# ---

# In[26]:


# Splitting the data into training and testing sets for diabetes balanced

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)


# ---
# ### 5. Feature Scaling
# ---

# In[27]:


# Feature Scaling on diabetes dataset

scaler_d = StandardScaler()
X_train_scaled = scaler_d.fit_transform(X_train)
X_test_scaled = scaler_d.transform(X_test)


# <h3 
#     align="center" 
#     style="font-size: 35px; padding:20px; font-family:Georgia; text-align:center; display:block; border-radius:20px;background-color:#454545">
#     <span style="color: white;">🎯 Model Building 🎯 </span>
# </h3>

# In[28]:


# Create the models

lr_d = LogisticRegression()
rf_d = RandomForestClassifier()

# Fit the models

lr_d.fit(X_train_scaled, y_train)
rf_d.fit(X_train_scaled, y_train)

# Make predictions

lr_pred_d = lr_d.predict(X_test_scaled)
rf_pred_d = rf_d.predict(X_test_scaled)


# <h3 
#     align="center" 
#     style="font-size: 35px; padding:20px; font-family:Georgia; text-align:center; display:block; border-radius:20px;background-color:#454545">
#     <span style="color: white;">⚡ Model Evaluation ⚡ </span>
# </h3>

# In[29]:


def plot_classification_report(report, title):
    lines = report.split('\n')[2:-5]
    classes = []
    precision = []
    recall = []
    f1_score = []
    support = []
    for line in lines:
        row_data = line.split()
        classes.append(row_data[0])
        precision.append(float(row_data[1]))
        recall.append(float(row_data[2]))
        f1_score.append(float(row_data[3]))
        support.append(int(row_data[4]))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=classes, y=precision, name='Precision', marker_color = colors[0]))
    fig.add_trace(go.Bar(x=classes, y=recall, name='Recall', marker_color = colors[1]))
    fig.add_trace(go.Bar(x=classes, y=f1_score, name='F1-Score', marker_color = colors[2]))

    fig.update_layout(title=title, xaxis_title='Class', yaxis_title='Score', barmode='group', xaxis={'categoryorder': 'total descending'}, plot_bgcolor='white')

    fig.show()


# In[30]:


# Classification reports for different algorithms

lr_d_report = classification_report(y_test, lr_pred_d)

rf_d_report = classification_report(y_test, rf_pred_d)


# In[31]:


# Plot classification reports

print("="*40, "Logistic regression report:", "="*45, '\n')
print(lr_d_report)
plot_classification_report(lr_d_report, "Logistic Regression Classification Report Visualization")


print("="*40, "Random forest report:", "="*45, '\n')
print(rf_d_report)
plot_classification_report(rf_d_report, "Random Forest Classification Report")


# In[32]:


# Calculate ROC and AUC for each model

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_pred_d)
lr_auc = auc(lr_fpr, lr_tpr)

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_d)
rf_auc = auc(rf_fpr, rf_tpr)

# Create ROC curve plot using Plotly

fig = go.Figure()
fig.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, mode='lines', name=f'Logistic Regression (AUC = {lr_auc:.2f})', line=dict(color=colors[1])))
fig.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode='lines', name=f'Random Forest (AUC = {rf_auc:.2f})', line=dict(color=colors[3])))

fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve for Diabetes', xaxis=dict(title='False Positive Rate'), yaxis=dict(title='True Positive Rate'), legend=dict(x=0.7, y=0.2), autosize=False, width=900, height=500, plot_bgcolor='white')
fig.show()


# <h3 
#     align="center" 
#     style="font-size: 35px; padding:20px; font-family:Georgia; text-align:center; display:block; border-radius:20px;background-color:#454545">
#     <span style="color: white;">🎈 Conclusion 🎈 </span>
# </h3>

# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=black>Conclusion:</font></h3>
#     
# ---    
# In conclusion, this notebook has provided valuable insights into the risk factors and predictive measures for heart diseases. Through meticulous data analysis, feature engineering, and advanced machine learning techniques, we have uncovered significant associations between various factors and disease occurrences.
# 
#  ---   
# **Heart Disease Insights:**
#          
#     
# In conclusion, the analysis of the dataset yielded several valuable insights. Null values were effectively handled, and data preprocessing included changes in column names, values, and data types. A new feature, 'bmi_group,' was created based on BMI bins and underwent data type and position adjustments. The overall health profile of the dataset indicated that most individuals were in good health, with normal weight and recent checkups, along with regular exercise. Furthermore, key factors contributing to heart disease were identified, including poor health, higher BMI, gender, age group, smoking, alcohol consumption, depression, and diabetes. Additionally, machine learning analysis using Random Forest demonstrated the best performance with an impressive F1-score and AUC of 0.93, suggesting its effectiveness in predicting heart disease in this context.
# 
# ---    
#     
# **What's next?** 
#     
# These discoveries emphasize the significance of personalized interventions and risk evaluation within public health strategies. By integrating these observations into clinical procedures, we can elevate disease prevention and treatment, ultimately leading to enhanced health outcomes on a broader scale.
#     
# ---

# ***Thankyou so much for veiwing the notebook. If you liked the notebook, please upvote and give a feedback to improve it further ❤️***

# ![360_F_291522205_XkrmS421FjSGTMRdTrqFZPxDY19VxpmL.jpg](attachment:4a97e277-7790-4ffb-ad88-78e739e0f76f.jpg)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




