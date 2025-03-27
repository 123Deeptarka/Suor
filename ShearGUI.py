# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:40:03 2025

@author: deeptarka.roy
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:08:51 2024

@author: deeptarka.roy
"""

#GUI for Thesis 
import pandas as pd 
import streamlit as st 
import numpy as np 
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
#rom tensorflow.keras.models import Sequential
#rom tensorflow.keras.layers import Conv1D,Activation,MaxPooling1D,Dense,Flatten
#mport keras 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score


st.markdown('<h1 style="font-size: 40px; font-weight: bold;"> Pushover Curve and Damage States</h1>', unsafe_allow_html=True)

st.sidebar.header("Specify Input Parameters")


#import data

df = pd.read_excel('shearmldata.xlsx')
x = df[["D","L/D","fc","fyl","fyt","pl","pt","Ny"]]
y = df[["DS1","DS2","DS3","DS4","F1","F2","F3","F4"]]

y_DS1=df[["DS1"]]
y_DS2=df[["DS2"]]
y_DS3=df[["DS3"]]
y_DS4=df[["DS4"]]
y_F1=df[["F1"]]
y_F2=df[["F2"]]
y_F3=df[["F3"]]
y_F4=df[["F4"]]





from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)



model=XGBRegressor(n_estimators=100,random_state=0,max_depth=5,max_leaves=20,reg_lambda=1,reg_alpha=2)
#model=XGBRegressor()
#model=RandomForestRegressor()
#model=LinearRegression()
model.fit(x_train,y_train)
#pred=model.predict(x_test)

y_pred=model.predict(x_test)
y_pred_train=model.predict(x_train)


print("The R2 value for Test Set for LR is :",r2_score(y_pred,y_test))

def user_input_features():
    D=st.sidebar.number_input("D",value=700.000,format="%.3f")
    LD =st.sidebar.number_input("L/D",value=3.500,format="%.3f")   
    fc =st.sidebar.number_input("fc",value=45.000,format="%.3f")
    fyl =st.sidebar.number_input("fyl",value=450.000,format="%.3f")
    fyt =st.sidebar.number_input("fyt",value=450.000,format="%.3f")
    pl =st.sidebar.number_input("pl",value=0.0190,format="%.3f")
    pt =st.sidebar.number_input("pt",value=0.00190,format="%.4f")
    Ny =st.sidebar.number_input("Ny",value=0.550,format="%.3f")
    data={"D":D,"L/D":LD,"fc":fc,"fyl":fyl,"fyt":fyt,"pl":pl,"pt":pt,"Ny":Ny}
    features=pd.DataFrame(data,index=[0])
    features_round=features.round(4)
    return features_round

Data=user_input_features()
prediction=model.predict(Data)

st.header("Specified Input Parameters")

new_column_names = {
    "D": "D (mm)",
    "LD": "L/D",
    "fc": "fc (MPa)",
    "fyl": "fyl (MPa)",
    "fyt": "fyt (MPa)",
    "pl": "pl", 
    "pt": "pt",
    "Ny": "Ny"
}

# Rename the columns
Data.rename(columns=new_column_names, inplace=True)

#style = [
#   dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848")]),
#    dict(selector="td", props=[("font-size", "16px"),("font-weight", "bold") ,("color", "#484848")])
#]
style = [
    dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848"), ("border", "4px solid #484848")]),
    dict(selector="td", props=[("font-size", "16px"), ("font-weight", "bold"), ("color", "#484848"), ("border", "4px solid #484848")]),
    dict(selector="table", props=[("border-collapse", "collapse")])  # Ensures borders are collapsed
]
Data_New=pd.DataFrame(Data,columns=["D(mm)","L/D","fc(Mpa)","fyl(Mpa)","fyt(Mpa)","pl","pt","Ny"])
# Apply styling to dataframe
st_df = Data.style.set_table_styles(style).format("{:.3f}").hide(axis="index")
# Convert to HTML without index and display
# Reset the index and drop it to remove the index column
Data_no_index = Data.reset_index(drop=True)

# Convert the DataFrame to HTML without the index
html = Data_no_index.to_html(index=False)

# Apply custom CSS to the HTML table
html = f"""
<style>
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th, td {{
        border: 6px solid #484848 ;
        padding: 3px;
        text-align: left;
    }}
    th {{
        font-size: 20px;
        font-weight: bold;
        color: #484848;
        border:2px solid #484848;
    }}
    td {{
        font-size: 16px;
        font-weight: bold;
        color: #484848;
    }}
</style>
{html}
"""

# Display the HTML table
st.markdown(html, unsafe_allow_html=True)





# Display using st.table or st.dataframe
#st.table(st_df)





st.header("Predicted Damage States ")

P=pd.DataFrame(prediction,columns=["DS1","DS2","DS3","DS4","F1 (kN)","F2 (kN)","F3 (kN)","F4 (kN)"])

P_DS=P[["DS1","DS2","DS3","DS4"]]


#st.dataframe(P_DS,hide_index=True)
P_F=P[["F1 (kN)","F2 (kN)","F3 (kN)","F4 (kN)"]]
#P_DS_display = P_DS.reset_index(drop=True)
#st.write(P_DS)
#styles = [
 #   dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848")]),
 #   dict(selector="td", props=[("font-size", "16px"),("font-weight", "bold") ,("color", "#484848")])
#]
# Apply styling to dataframe
#styled_df = P_DS.style.set_table_styles(styles)
#st.table(styled_df)

# Title
#st.write("Drift Ratio")

# Title with Markdown for styling
st.markdown("<h1 style='text-align: center; font-size: 20px; font-weight: bold; color: #484848;'>Drift Ratio (%)</h1>", unsafe_allow_html=True)

# Subtitles
#st.write("DS1, DS2, DS3, DS4")

# Styling
#styles = [
   # dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848")]),
   # dict(selector="td", props=[("font-size", "16px"), ("font-weight", "bold")    ,("color", "#484848")])
#]
styles = [
    dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848"), ("border", "4px solid #484848")]),
    dict(selector="td", props=[("font-size", "16px"), ("font-weight", "bold"), ("color", "#484848"), ("border", "4px solid #484848")]),
    dict(selector="table", props=[("border-collapse", "collapse")])  # Ensures borders are collapsed
]

# Apply styling to dataframe
styled_df = P_DS.style.set_table_styles(styles).format("{:.3f}").hide(axis="index")

#html = styled_df.to_html(index=False)
#st.write(html,unsafe_allow_html=True)
# Display the table
#st.table(styled_df)
P_DS_no_index = P_DS.round(2).reset_index(drop=True)
html = P_DS_no_index.to_html(index=False)

# Apply custom CSS to the HTML table
html = f"""
<style>
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th, td {{
        border: 3px solid black !important;  /* Thicker border with !important */
        padding: 4px;
        text-align: center;
    }}
    th {{
        font-size: 20px;
        font-weight: bold;
        color: #484848;
        border:4px solid #484848;
    }}
    td {{
        font-size: 16px;
        font-weight: bold;
        color: #484848;
        border:4px solid #484848;
    }}
</style>
{html}
"""

# Display the HTML table
st.markdown(html, unsafe_allow_html=True)






Outpred_DR_0=P_DS.to_numpy().reshape(4,)

Outpred_F_0=P_F.to_numpy().reshape(4,)


# In[20]:


a=np.insert(Outpred_DR_0,0,0)
b=np.insert(Outpred_F_0,0,0)

# In[21]:
st.header("Predicted Pushover Curve ")
fig,ax=plt.subplots(figsize=(6,3))
ax.plot(a,b,label="Predicted Pushover Curve",marker="o")
#ax.plot(a1,b1,label="Simulated Pushover Curve",marker="o")
ax.set_xlabel("Drift Ratio (%)")
ax.set_ylabel("Force (kN)")
#ax.set_title("Predicted VS Simulated Pushover Curves")
#ax.legend()
#ax.show()
#st.pyplot(fig)
# Label the points
for i in range(1, 5):  # We start from 1 to skip the (0,0) point
    ax.annotate(f'DS{i}', (a[i], b[i]), textcoords="offset points", xytext=(5,-20), ha='center')

# Add grid for better readability
#ax.grid(True, linestyle='--', alpha=0.7)

# Use tight layout to prevent clipping of labels
plt.tight_layout()

st.pyplot(fig, use_container_width=True)