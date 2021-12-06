import streamlit as st

st.markdown('# Final Project')
st.markdown('Name: Yuzhen Lang')
st.markdown('Student ID: 53364275')
st.markdown('## Link')
st.markdown('[Link to the dataset](https://data.cnra.ca.gov/dataset/3f96977e-2597-4baa-8c9b-c433cea0685e/resource/24fc759a-ff0b-479a-a72a-c91a9384540f/download/stations.csv)')
st.markdown('[Link to the GitHub](https://github.com/Langyuzhen)')
st.markdown('## Import: The Dataset is about Waterquality')

import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
rng = np.random.default_rng()

df = pd.read_csv("Waterquality.csv")
st.dataframe(df)
st.markdown('## Basic Information: Using Pandas to Explore the Dataset')
st.markdown('#### Shape of df:')
st.dataframe(df.shape)
st.markdown('#### Columns of df:')
st.markdown(df.columns)
st.markdown('#### Check for null')
st.dataframe(df.notna())
st.dataframe((~df.isna().all(axis = 0)))
st.markdown('#### Statistical Description')
st.dataframe(df.describe())
st.markdown('#### Stations where sample count is 1:')
st.dataframe(df.loc[df['SAMPLE_COUNT'] == 1])
st.markdown('#### The Station where sample count is max:')
st.dataframe(df.loc[df['SAMPLE_COUNT'] == 1631])

st.markdown('## Classificating the stations with latitude and longitude')
st.markdown('As every station belongs to a county, county name is a label for the station. With latitude and longitude of stations, we can classify the stations into counties using KNN or K-mans.')
x1=df['LONGITUDE']
y1=df['LATITUDE']
st.markdown('This is the plot of longtitude(as x) and latitude(as y) of stations:')
fig1=plt.figure()
plt.scatter(x1,y1)
st.pyplot(fig1)
st.markdown('There are more than forty thousand points in this dataset. As is shown in the picture, it is too many for the algorithm. And points are too dense to classify. So I choose 400 entries randomly from the dataset to classify')
st.markdown('##### We choose 400 entries from the data randomly and plot again:')
np.random.seed(4)
dff=df.sample(400)
dfrd=dff
st.dataframe(dfrd)
x2=dfrd['LONGITUDE']
y2=dfrd['LATITUDE']
plt.scatter(x2,y2)
fig2=plt.figure()
plt.scatter(x2,y2)
st.pyplot(fig2)
st.altair_chart(alt.Chart(dfrd).mark_point().encode(
    x=alt.X('LONGITUDE', scale=alt.Scale(domain=[-125,-114])),
    y=alt.Y('LATITUDE', scale=alt.Scale(domain=[32,43])),
    color='COUNTY_NAME'
))
st.markdown('##### county names of stations:')
st.dataframe(set(dfrd['COUNTY_NAME']))
st.markdown('How many different counties in this dataset:')
st.text(len(set(dfrd['COUNTY_NAME'])))

st.markdown('## K-means (Unsupervised Learning)')
st.markdown('#### Steps')
st.markdown('Assuming we have input data points $x_1,x_2,x_3,...,x_n$ and value of K (the number of clusters needed, also as the number of clusters needed where in this case we have K=51). We follow the below procedure:')
st.markdown('1.Pick K points as the initial centroids from the dataset, either randomly or the first K.')
st.markdown('2.Find the Euclidean distance of each point in the dataset with the identified K points (cluster centroids).')
st.markdown('3.Assign each data point to the closest centroid using the distance found in the previous step.')
st.markdown('4.Find the new centroid by taking the average of the points in each cluster group.')
st.markdown('5.Repeat 2 to 4 for a fixed number of iteration or till the centroids don’t change.')
st.markdown('#### Details')
st.markdown('We have to decide the method of calculating the distance between two points, which is usually Euclidean Distance. As in this case there are two location parameters for one point(longitude and latitude), it is best to choose the Euclidean Distance:')
st.markdown('If$\mathbf{p}=(p_1,p_2)$ and $\mathbf{q}=(q_1,q_2)$ then the distance is given by')
st.markdown('$$d(\mathbf{p},\mathbf{q})=\sqrt{(p_1-q_1)^2+(p_2-q_2)^2}.$$')
st.markdown('If each cluster centroid is denoted by $c_i$, then each data point x is assigned to a cluster based on')
st.markdown('$$argmin_{c_i\in C}d(c_i,x)^2$$')
st.markdown('Fingding the new centroid from the clustered group of points:')
st.markdown('$$c_i=\sum_{x_i\in S_i}{x_i}/|S_i|$$')
st.markdown('$S_i$ is the set of all points assigned to the $i$th cluster and $|S_i|$ is the number of points in $S_i$.')     

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

K = st.slider('Please decide the parameter K',min_value=1,max_value=100,value=5)
K=int(K)
X=dfrd.loc[:,'LATITUDE':'LONGITUDE']
k_means=KMeans(K)
k_means.fit(X)
cluster_centers=k_means.cluster_centers_
y_kmeans=k_means.predict(X)
fig3=plt.figure()
plt.scatter(X.loc[:,'LONGITUDE'],X.loc[:,'LATITUDE'],c=y_kmeans,s=50,cmap='viridis')
st.pyplot(fig3)
st.markdown('This portion of the code was partly taken from https://muthu.co/mathematics-behind-k-mean-clustering-algorithm/ ')

st.markdown('## Logistic Regression (Supervised Learning)')
st.markdown('Assuming we have input data points $x_1,x_2,x_3,...,x_n$. We will model the probability using a function of the form') 
st.markdown('$$1/({1+e^{a_0+a_1x_1+...+a_nx_n}})$$')
st.markdown('The goal of linear regression is to select the best possible coefficients $a_0,a_1,...,a_n$.')

st.markdown('The function $$f(x)=1/({1+e^{-x}})$$ is called sigma function, also known as logistic function. This function has many important properties such as :')
st.markdown('$$\lim_{x to - \infty}f(x)=0,\quad \lim_{x to  \infty}f(x)=1,$$')
st.markdown('$f(x)$ is increasing through the $\mathbf{R}$ ,$f(x)$ is derivable infinitely.')

import math 

xm=list(np.arange(-10,10,0.01))
ym=[]
for i in range(len(xm)):
    ym.append(math.exp(xm[i])/(1+math.exp(xm[i])))
xm=pd.DataFrame(xm,columns=['xx'])
ym=pd.DataFrame(ym,columns=['yy'])
xym=pd.concat([xm,ym],axis=1)
st.altair_chart(alt.Chart(xym).mark_point().encode(
    x='xx',y='yy'
))
st.markdown('According to the properties, it is a good candidate for modeling probability, because its outputs are always in correct range, between 0 and 1.')
st.markdown('How do we decide what coefficients $a_0,a_1,...,a_n$ best model the data? We will use a cost function (or loss function). The smaller the value of the cost function, the better the fit. ')
st.markdown('Say we have n data points, and for each of those we have an actual output y, where y is always 0 or 1. In addition to the actual output, we have our model’s output, which is a probability. Here is a naive first guess for the cost function:')
st.markdown('$$1/n*(\sum_{i=1}{n}-\log p(y^{(i)}))$$')
st.markdown('where $p(y^{(i)})$ denotes the predicted probability of $y^{(i)}$.')
st.markdown('This naive cost function has the good quality that if we perfectly predict every outcome (meaning we give the actual outcome probability 1 every time), then the cost function is 0. So the best possible predictions lead to the best possible score. The problem is this cost function does not suitably punish predictions which are both confident and wrong.')
county1=dict(enumerate(set(dfrd['COUNTY_NAME'])))
county2={}

st.markdown('##### Using train_test_split to split the data into training and testing sets.')
for key,val in county1.items():
    county2[val]=key
for key,val in county2.items():
    dfrd.replace(key,val,inplace=True)

from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test = train_test_split(dfrd.loc[:,'LATITUDE':'LONGITUDE'],dfrd.loc[:,'COUNTY_NAME'],train_size=0.8)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
aaa=model.score(X_test,Y_test)
st.markdown('##### The accuracy ratio of the model is:')
st.text(aaa)
st.markdown('While the result is not as good as we expect, for this case we have to choose K-Means to classify the county names.')
