import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

#Question 1 
st.title("Math 10 Homework 4")

#Question 2
st.markdown("Langyuzhen, Grace Lee, Emi Cervantes")

#Question 3 free to  uploaded file in CSV format
uploaded_file = st.file_uploader(label = "Upload a CSV file, please", type = "CSV")

#Question 4 As pandas already imported as pd
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

#Question 5 Using applymap and lambda function to replace blank space
    df = df.applymap(lambda x: np.nan if x == "" else x)

#Question 6 first define a function  
    def can_be_numeric(c):
        try:
            pd.to_numeric(df[c])
            return True
        except:
            return False
 # Using list comprehension to conclude columns in dataframe that can be numeric
    good_cols = [c for c in df.columns if  can_be_numeric(c)]

#Question 7
    df[good_cols] = df[good_cols].apply(pd.to_numeric, axis = 0)

#Question 8
    x_axis = st.selectbox("Choose an x-value", good_cols)
    y_axis = st.selectbox("Choose an y-value", good_cols)
    
#Question 9
    values = st.slider("Select the range of rows:", 0,len(df.index)-1,(0,len(df.index)))

#Question 10    
    st.write(f"Plotting {values}")

#Question 11
    my_graph = alt.Chart(df.loc[values[0]:values[1]]).mark_circle().encode(
        x = x_axis, 
        y = y_axis,
        color = x_axis, #Q12   
    )
    st.altair_chart(my_graph)

#Question 12 Add one more funny operator in myapp
    number = st.number_input("Insert a number", 0)
    st.write("Choose a number you like", x_axis)
