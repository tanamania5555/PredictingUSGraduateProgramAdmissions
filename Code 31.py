#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib notebook
import pandas as pd
import numpy as np
import os
import sys
import time
import pickle
import seaborn
from scipy import stats
from sklearn import svm, preprocessing
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, f1_score
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("dataforproject.csv")
print(df.columns)
df


# In[3]:


df.dropna()


# In[4]:


# df = df[~(df['UG College'] == 'NaN')]
df = df[pd.notnull(df['UG College'])]
df


# In[5]:


df = df.fillna(0)
df


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[8]:


le = preprocessing.LabelEncoder()
df['UG label'] = le.fit_transform(df['UG College'].astype(str))
df


# In[9]:


df = df.drop(columns=['Username', 'Specialization', 'Major', 'Department', 'User Profile Link', 'Term & Year', 'UG College', 'gmatV', 'gmatQ', 'gmatA'])
df


# In[10]:


df.describe()


# In[11]:


l = len(df)
df = df[~(df['CGPA Scale'] == 0)]
l1 = len(df)
print(l-l1, "outliers removed.")

l = len(df)
df = df[~(df['CGPA'] == 0)]
l1 = len(df)
print(l-l1, "outliers removed.")


# In[12]:


df['CGPA'] = 10*df['CGPA']/df['CGPA Scale']
df


# In[13]:


df['Topper CGPA'] = 10*df['Topper CGPA']/df['CGPA Scale']
df


# In[14]:


df['Topper CGPA'] = df['Topper CGPA'].apply(lambda x: 9 if float(x)==0 else float(x))
df


# In[15]:


df = df.drop(columns=['CGPA Scale'])
df


# In[16]:


# Outlier Removal
l = len(df)
df = df[~(df['CGPA'] <= 3)]
l1 = len(df)
print(l-l1, 'outliers removed.')
# df.reset_index(inplace=True)


# In[17]:


# Outlier Removal
l = len(df)
df = df[~(df['CGPA'] > 10)]
l1 = len(df)
print(l-l1, 'outliers removed.')
# df.reset_index(inplace=True)


# In[18]:


# Outlier Removal
l = len(df)
df = df[~(df['Topper CGPA'] > 10)]
l1 = len(df)
print(l-l1, 'outliers removed.')
# df.reset_index(inplace=True)


# In[19]:


# Outlier Removal
l = len(df)
df = df[~(df['Topper CGPA'] <= 5.5)]
l1 = len(df)
print(l-l1, 'outliers removed.')
# df.reset_index(inplace=True)


# In[20]:


mean_greA = df['greA'].mean()
print(mean_greA)
df['greA'] = df['greA'].apply(lambda x: mean_greA if x == 0 else x)
df


# In[21]:


mean_greQ = df['greQ'].mean()
print(mean_greQ)
df['greQ'] = df['greQ'].apply(lambda x: mean_greQ if x == 0 else x)
df


# In[22]:


mean_greV = df['greV'].mean()
print(mean_greV)
df['greV'] = df['greV'].apply(lambda x: mean_greV if x == 0 else x)
df


# In[23]:


def func0(program):
    if program.upper() == 'MS':
        return 0
    elif program.upper() == 'PHD':
        return 1
    else:
        return 2
df['Program'] = df['Program'].apply(func0)
df


# In[24]:


# def func1(name):
#     name = name.upper()
#     if name == 'IIT':
#         return 0
#     elif name == 'IIIT':
#         return 1
#     elif name == 'NIT':
#         return 2
#     else:
#         return 3
# df['ugCollege'] = df['ugCollege'].apply(func1)
# df


# In[25]:


univ=['Carnegie Mellon University',
       'University of North Carolina Chapel Hill',
       'University of Illinois Urbana-Champaign',
       'University of California San Diego',
       'University of Minnesota Twin Cities',
       'Texas A and M University College Station',
       'Georgia Institute of Technology', 'University of Texas Austin',
       'University of Michigan Ann Arbor', 'Columbia University',
       'University of Maryland College Park', 'Arizona State University',
       'University of Cincinnati', 'Ohio State University Columbus',
       'North Carolina State University', 'Northeastern University',
       'University of Arizona', 'University of Wisconsin Madison',
       'SUNY Buffalo', 'Clemson University', 'University of Utah',
       'Rutgers University New Brunswick/Piscataway',
       'Virginia Polytechnic Institute and State University',
       'Stanford University', 'Massachusetts Institute of Technology',
       'California Institute of Technology',
       'University of Massachusetts Amherst',
       'University of California Irvine', 'Purdue University',
       'Cornell University', 'University of Florida',
       'University of Washington', 'Syracuse University',
       'University of Pennsylvania', 'University of Southern California',
       'University of Texas Dallas', 'University of Illinois Chicago',
       'George Mason University', 'Harvard University',
       'Johns Hopkins University', 'SUNY Stony Brook',
       'Northwestern University', 'New York University',
       'New Jersey Institute of Technology',
       'University of California Santa Barbara', 'Princeton University',
       'University of Colorado Boulder',
       'University of California Los Angeles',
       'University of North Carolina Charlotte',
       'University of Texas Arlington', 'University of California Davis',
       'Worcester Polytechnic Institute',
       'University of California Santa Cruz', 'Wayne State University']
ranks = [48,90,75,45,156,189,72,65,21,18,136,215,561,101,285,344,262,56,340,701,353,262,327,2,1,5,305,219,111,14,167,68,581,15,129,501,231,801,3,24,359,31,39,751,135,13,206,35,90,301,104,601,367,484]
print(len(univ), len(ranks))
univdict = {univ[i]: ranks[i] for i in range(len(univ))} 
print(univdict)


# In[26]:


ranking = []
# uniqueUnivs = list(df['University Name'].unique())
# print((uniqueUnivs))
for index, row in df.iterrows():
    # i = uniqueUnivs.index(row['University Name'])
    # print(row['University Name'])
    ranking.append(univdict[row['University Name']])
print(len(ranking), len(ranks))
df['ranking'] = ranking
df


# In[27]:


df = df.drop(columns='University Name')
df


# In[28]:


df.reset_index(inplace=True)
df = df.drop(columns=['index'])
df


# In[29]:


def func2(x):
    if x > 0 and x < 101:
        return 0
    elif x > 100 and x < 251:
        return 1
    elif x > 250 and x < 401:
        return 2
    else:
        return 3

flag = 0
if flag == 1:
    df["rank"] = df["ranking"].apply(func2)
df


# In[30]:


df_y = df['Admission']

# To check whether a column has very uneven distribution of class of y (0 or 1)
for i in range(len(df.columns)):
    plt.figure(i, figsize=(25,2))
    plt.title(df.columns[i] + ' vs admission chances')
    plt.scatter(df.iloc[:, i], df_y, c='blue', label=df.columns[i], alpha=0.5)
    plt.xlabel(df.columns[i] + ' (x)')
    plt.ylabel('Rate of admission (y)')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[31]:


df['z'] = np.abs(stats.zscore(df['greQ']))
# print(df['z'].describe())
df['z'].to_csv(r'test/test_greQ.csv')

# Outlier Removal
l = len(df)
df = df[~(df['z'] >= 2)]
l1 = len(df)
print(l-l1, 'outliers removed.')
df = df.drop(columns='z')


# In[32]:


df['z'] = np.abs(stats.zscore(df['greV']))
# print(df['z'].describe())
df['z'].to_csv(r'test/test_greV.csv')

# Outlier Removal
l = len(df)
df = df[~(df['z'] >= 2)]
l1 = len(df)
print(l-l1, 'outliers removed.')
df = df.drop(columns='z')


# In[33]:


df['z'] = np.abs(stats.zscore(df['greA']))
# print(df['z'].describe())
df['z'].to_csv(r'test/test_greA.csv')

# Outlier Removal
l = len(df)
df = df[~(df['z'] >= 0.1)]
l1 = len(df)
print(l-l1, 'outliers removed.')
df = df.drop(columns='z')


# In[34]:


df['z'] = np.abs(stats.zscore(df['TOEFL Essay']))
# print(df['z'].describe())
df['z'].to_csv(r'test/test_TOEFL Essay.csv')

# Outlier Removal
l = len(df)
df = df[~(df['z'] >= 2.5)]
l1 = len(df)
print(l-l1, 'outliers removed.')
df = df.drop(columns='z')


# In[35]:


df['z'] = np.abs(stats.zscore(df['TOEFL Score']))
# print(df['z'].describe())
df['z'].to_csv(r'test/test_TOEFL Score.csv')

# Outlier Removal
l = len(df)
df = df[~(df['z'] >= 4.6)]
l1 = len(df)
print(l-l1, 'outliers removed.')
df = df.drop(columns='z')


# In[36]:


df['z'] = np.abs(stats.zscore(df['Journal Pubs']))
# print(df['z'].describe())
df['z'].to_csv(r'test/test_Journal Pubs.csv')

# Outlier Removal
l = len(df)
df = df[~(df['z'] >= 4)]
l1 = len(df)
print(l-l1, 'outliers removed.')
df = df.drop(columns='z')


# In[37]:


df['z'] = np.abs(stats.zscore(df['Intern Exp']))
# print(df['z'].describe())
df['z'].to_csv(r'test/test_Intern Exp.csv')

# Outlier Removal
l = len(df)
df = df[~(df['z'] >= 3)]
l1 = len(df)
print(l-l1, 'outliers removed.')
df = df.drop(columns='z')


# In[38]:


df['z'] = np.abs(stats.zscore(df['Industry Exp']))
# print(df['z'].describe())
df['z'].to_csv(r'test/test_Industry Exp.csv')

# Outlier Removal
l = len(df)
df = df[~(df['z'] >= 4.2)]
l1 = len(df)
print(l-l1, 'outliers removed.')
df = df.drop(columns='z')


# In[39]:


df_y = df['Admission']
df = df.drop(columns='Admission')


# In[42]:


# To check whether a column has very uneven distribution of class of y (0 or 1) AGAIN!
for i in range(len(df.columns)):
    plt.figure(i, figsize=(25,2))
    plt.title(df.columns[i] + ' vs Admission Chance')
    plt.scatter(df.iloc[:, i], df_y, c='blue', label=df.columns[i], alpha=0.5)
    plt.xlabel(df.columns[i] + ' (x)')
    plt.ylabel('Rate of admission (y)')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[43]:


df.to_csv(r'dataframe_preprocessed.csv')
df


# In[44]:


df['Admission'] = df_y

print(df.shape)
fig, ax = plt.subplots(figsize=(9, 5))
plt.title('Correlation between different features', fontsize=8)
ax.title.set_position([0.5, 1.05])
# ax.axis('off')
seaborn.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt='.2f',cmap="magma")
plt.show()


# In[45]:


df = shuffle(df)
df.reset_index(inplace=True)
df_y = df['Admission']
df = df.drop(columns=['index', 'Admission'])
df


# In[47]:


df.replace([np.inf, -np.inf], np.nan).dropna()


# In[48]:


l = len(df)

dfa = np.array(df[:(l//5)])
dfb = np.array(df[(l//5):(l*2)//5])
dfc = np.array(df[((2*l)//5):(l*3)//5])
dfd = np.array(df[((3*l)//5):(l*4)//5])
dfe = np.array(df[((4*l)//5):])

df_ya = np.array(df_y[:(l//5)])
df_yb = np.array(df_y[(l//5):(l*2)//5])
df_yc = np.array(df_y[((2*l)//5):(l*3)//5])
df_yd = np.array(df_y[((3*l)//5):(l*4)//5])
df_ye = np.array(df_y[((4*l)//5):])

dfbcde = np.concatenate((dfb,dfc,dfd,dfe))
dfacde = np.concatenate((dfa,dfc,dfd,dfe))
dfabde = np.concatenate((dfa,dfb,dfd,dfe))
dfabce = np.concatenate((dfa,dfb,dfc,dfe))
dfabcd = np.concatenate((dfa,dfb,dfc,dfd))

df_ybcde = np.concatenate((df_yb,df_yc,df_yd,df_ye))
df_yacde = np.concatenate((df_ya,df_yc,df_yd,df_ye))
df_yabde = np.concatenate((df_ya,df_yb,df_yd,df_ye))
df_yabce = np.concatenate((df_ya,df_yb,df_yc,df_ye))
df_yabcd = np.concatenate((df_ya,df_yb,df_yc,df_yd))

df_n = [dfbcde, dfacde, dfabde, dfabce, dfabcd]
df_y_n = [df_ybcde, df_yacde, df_yabde, df_yabce, df_yabcd]


# In[49]:


print(len(dfa), len(dfb), len(dfc), len(dfd), len(dfe))
print(len(df_ya), len(df_yb), len(df_yc), len(df_yd), len(df_ye))
print(len(dfbcde), len(dfacde), len(dfabde), len(dfabce), len(dfabcd))
print(len(df_ybcde), len(df_yacde), len(df_yabde), len(df_yabce), len(df_yabcd))


# In[53]:


# 5 folds
n_folds = 5

flag = 0
if flag == 1:
    
    knn1 = [KNeighborsClassifier(2)]*(n_folds)
    for i in range(n_folds):
        start = time.time()
        knn1[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('knn1' + str(i+1) + ' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/knn1-'+str(n_folds), 'wb') as file:
            pickle.dump(knn1, file)
    
    
    knn2 = [KNeighborsClassifier(3)]*(n_folds)
    for i in range(n_folds):
        start = time.time()
        knn2[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('knn2' + str(i+1) + ' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/knn2-'+str(n_folds), 'wb') as file:
            pickle.dump(knn2, file)
    
    
    rfc1 = [RandomForestClassifier(n_estimators=10)]*(n_folds)
    for i in range(n_folds):
        start = time.time()
        rfc1[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('rfc1'+str(i+1)+' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/rfc1-'+str(n_folds), 'wb') as file:
            pickle.dump(rfc1, file)
    
    
    rfc2 = [RandomForestClassifier(n_estimators=15)]*(n_folds)
    for i in range(n_folds):
        start = time.time()
        rfc2[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('rfc2' + str(i+1) + ' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/rfc2-'+str(n_folds), 'wb') as file:
            pickle.dump(rfc2, file)
    
    
    rfc3 = [RandomForestClassifier(n_estimators=1000)]*(n_folds)
    for i in range(n_folds):
        start = time.time()
        rfc3[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('rfc3' + str(i+1) + ' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/rfc3-'+str(n_folds), 'wb') as file:
            pickle.dump(rfc3, file)
    
    
    mlp1 = [MLPClassifier(alpha=0.5, max_iter=1000)]*(n_folds)
    for i in range(n_folds):
        start = time.time()
        mlp1[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('mlp1' + str(i+1) + ' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/mlp1-'+str(n_folds), 'wb') as file:
            pickle.dump(mlp1, file)
    
    
    mlp2 = [MLPClassifier(alpha=0.5, max_iter=2000)]*(n_folds)
    for i in range(n_folds):
        start = time.time()
        mlp2[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('mlp2' + str(i+1) + ' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/mlp2-'+str(n_folds), 'wb') as file:
            pickle.dump(mlp2, file)
    
    
    logr1 = [LogisticRegression(penalty='l1',tol=0.01)]*(n_folds)
    for i in range(n_folds):
        start = time.time()
        logr1[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('logr1'+str(i+1)+' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/logr1-'+str(n_folds), 'wb') as file:
            pickle.dump(logr1, file)

    
    logr2 = [LogisticRegression(penalty='l2',tol=0.01)]*(n_folds)
    for i in range(n_folds):
        start = time.time()
        logr2[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('logr2' + str(i+1) + ' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/logr2-'+str(n_folds), 'wb') as file:
            pickle.dump(logr2, file)


    bnb1 = [BernoulliNB()]*(n_folds)
    for i in range(n_folds):    
        start = time.time()
        bnb1[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('bnb1'+str(i+1)+' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/bnb1-'+str(n_folds), 'wb') as file:
        pickle.dump(bnb1, file)
    
    
    cnb1 = [ComplementNB()]*(n_folds)
    for i in range(n_folds):    
        start = time.time()
        cnb1[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('cnb1'+str(i+1)+' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/cnb1-'+str(n_folds), 'wb') as file:
        pickle.dump(cnb1, file)

        
    gnb1 = [GaussianNB()]*(n_folds)
    for i in range(n_folds):    
        start = time.time()
        gnb1[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('gnb1'+str(i+1)+' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/gnb1-'+str(n_folds), 'wb') as file:
        pickle.dump(gnb1, file)
        

    mnb1 = [MultinomialNB()]*(n_folds)
    for i in range(n_folds):    
        start = time.time()
        mnb1[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('mnb1'+str(i+1)+' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/mnb1-'+str(n_folds), 'wb') as file:
        pickle.dump(mnb1, file)


    svm1 = [svm.SVC(kernel='poly', degree=1)]*(n_folds)
    for i in range(n_folds):    
        start = time.time()
        svm1[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('svm1' + str(i+1) + ' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/svm1-'+str(n_folds), 'wb') as file:
        pickle.dump(svm1, file)
    
    
#     svm2 = [svm.SVC(kernel='poly', degree=2)]*(n_folds)
#     for i in range(n_folds):
#         start = time.time()
#         svm2[i].fit(df_n[i],df_y_n[i])
#         end = time.time()
#         print('svm2'+str(i+1)+' - Total Time: %.4f s' % (end-start))
#     with open('pickle-files/svm2-'+str(n_folds), 'wb') as file:
#             pickle.dump(svm2, file)
    
    
#     svm3 = [svm.SVC(kernel='poly', degree=3)]*(n_folds)
#     for i in range(n_folds):
#         start = time.time()
#         svm3[i].fit(df_n[i],df_y_n[i])
#         end = time.time()
#         print('svm3'+str(n_folds)+' - Total Time: %.4f s' % (end-start))
#     with open('pickle-files/svm3-'+str(n_folds), 'wb') as file:
#             pickle.dump(svm3, file)


#     svm4 = [svm.SVC(kernel='poly', degree=4)]*(n_folds)
#     for i in range(n_folds):
#         start = time.time()
#         svm4[i].fit(df_n[i],df_y_n[i])
#         end = time.time()
#         print('svm4'+str(n_folds)+' - Total Time: %.4f s' % (end-start))
#     with open('pickle-files/svm4-'+str(n_folds), 'wb') as file:
#             pickle.dump(svm4, file)
    
    
    svm5 = [svm.SVC(kernel='rbf', gamma='scale')]*(n_folds)
    for i in range(n_folds):    
        start = time.time()
        svm5[i].fit(df_n[i],df_y_n[i])
        end = time.time()
        print('svm5'+str(i+1)+' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/svm5-'+str(n_folds), 'wb') as file:
            pickle.dump(svm5, file)
    
    
    svm6 = [svm.SVC(kernel='linear')]*(n_folds)
    for i in range(n_folds):
        start = time.time()
        svm6[i].fit(df_n[i],df_y_n[i]) 
        end = time.time()
        print('svm6'+str(i+1)+' - Total Time: %.4f s' % (end-start))
    with open('pickle-files/svm6-'+str(n_folds), 'wb') as file:
        pickle.dump(svm6, file) 

elif flag == 0:
    with open('pickle-files/svm1-'+str(n_folds), 'rb') as file:
        svm1 = pickle.load(file)    
#     with open('pickle-files/svm2-'+str(n_folds), 'rb') as file:
#         svm2 = pickle.load(file)
#     with open('pickle-files/svm3-'+str(n_folds), 'rb') as file:
#         svm3 = pickle.load(file)
#     with open('pickle-files/svm4-'+str(n_folds), 'rb') as file:
#         svm4 = pickle.load(file)
    with open('pickle-files/svm5-'+str(n_folds), 'rb') as file:
        svm5 = pickle.load(file)
    with open('pickle-files/svm6-'+str(n_folds), 'rb') as file:
        svm6 = pickle.load(file)
    with open('pickle-files/knn1-'+str(n_folds), 'rb') as file:
        knn1 = pickle.load(file)
    with open('pickle-files/knn2-'+str(n_folds), 'rb') as file:
        knn2 = pickle.load(file)
    with open('pickle-files/rfc1-'+str(n_folds), 'rb') as file:
        rfc1 = pickle.load(file)
    with open('pickle-files/rfc2-'+str(n_folds), 'rb') as file:
        rfc2 = pickle.load(file)
    with open('pickle-files/rfc3-'+str(n_folds), 'rb') as file:
        rfc3 = pickle.load(file)
    with open('pickle-files/mlp1-'+str(n_folds), 'rb') as file:
        mlp1 = pickle.load(file)
    with open('pickle-files/mlp2-'+str(n_folds), 'rb') as file:
        mlp2 = pickle.load(file)
    with open('pickle-files/logr1-'+str(n_folds), 'rb') as file:
        logr1 = pickle.load(file)
    with open('pickle-files/logr2-'+str(n_folds), 'rb') as file:
        logr2 = pickle.load(file)
    with open('pickle-files/bnb1-'+str(n_folds), 'rb') as file:
        bnb1 = pickle.load(file)
    with open('pickle-files/cnb1-'+str(n_folds), 'rb') as file:
        cnb1 = pickle.load(file)
    with open('pickle-files/gnb1-'+str(n_folds), 'rb') as file:
        gnb1 = pickle.load(file)
    with open('pickle-files/mnb1-'+str(n_folds), 'rb') as file:
        mnb1 = pickle.load(file)


# In[57]:


# Testing Accuracy:
        
svm1_predy = []
svm2_predy = []
svm3_predy = []
svm4_predy = []
svm5_predy = []
svm6_predy = []
knn1_predy = []
knn2_predy = []
rfc1_predy = []
rfc2_predy = []
rfc3_predy = []
mlp1_predy = []
mlp2_predy = []
logr1_predy = []
logr2_predy = []
bnb1_predy = []
cnb1_predy = []
gnb1_predy = []
mnb1_predy = []
        
for i,a in zip(range(n_folds), [dfa, dfb, dfc, dfd, dfe]):
    svm1_predy += [svm1[i].predict(a)]
#     svm2_predy += [svm2[i].predict(a)]
#     svm3_predy += [svm3[i].predict(a)]
#     svm4_predy += [svm4[i].predict(a)]
    svm5_predy += [svm5[i].predict(a)]
    svm6_predy += [svm6[i].predict(a)]
    knn1_predy += [knn1[i].predict(a)]
    knn2_predy += [knn2[i].predict(a)]
    rfc1_predy += [rfc1[i].predict(a)]
    rfc2_predy += [rfc2[i].predict(a)]
    rfc3_predy += [rfc3[i].predict(a)]
    mlp1_predy += [mlp1[i].predict(a)]
    mlp2_predy += [mlp2[i].predict(a)]
    logr1_predy += [logr1[i].predict(a)]
    logr2_predy += [logr2[i].predict(a)]
    bnb1_predy += [bnb1[i].predict(a)]
    cnb1_predy += [cnb1[i].predict(a)]
    gnb1_predy += [gnb1[i].predict(a)]
    mnb1_predy += [mnb1[i].predict(a)]

svm1_acc = []
svm2_acc = []
svm3_acc = []
svm4_acc = []
svm5_acc = []
svm6_acc = []
knn1_acc = []
knn2_acc = []
rfc1_acc = []
rfc2_acc = []
rfc3_acc = []
mlp1_acc = []
mlp2_acc = []
logr1_acc = []
logr2_acc = []
bnb1_acc = []
cnb1_acc = []
gnb1_acc = []
mnb1_acc = []

for i,fold in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    svm1_acc += [accuracy_score(fold, svm1_predy[i])]
#     svm2_acc += [accuracy_score(fold, svm2_predy[i])]
#     svm3_acc += [accuracy_score(fold, svm3_predy[i])]
#     svm4_acc += [accuracy_score(fold, svm4_predy[i])]
    svm5_acc += [accuracy_score(fold, svm5_predy[i])]
    svm6_acc += [accuracy_score(fold, svm6_predy[i])]
    knn1_acc += [accuracy_score(fold, knn1_predy[i])]
    knn2_acc += [accuracy_score(fold, knn2_predy[i])]
    rfc1_acc += [accuracy_score(fold, rfc1_predy[i])]
    rfc2_acc += [accuracy_score(fold, rfc2_predy[i])]
    rfc3_acc += [accuracy_score(fold, rfc3_predy[i])]
    mlp1_acc += [accuracy_score(fold, mlp1_predy[i])]
    mlp2_acc += [accuracy_score(fold, mlp2_predy[i])]
    logr1_acc += [accuracy_score(fold, logr1_predy[i])]
    logr2_acc += [accuracy_score(fold, logr2_predy[i])]
    bnb1_acc += [accuracy_score(fold, bnb1_predy[i])]
    cnb1_acc += [accuracy_score(fold, cnb1_predy[i])]
    gnb1_acc += [accuracy_score(fold, gnb1_predy[i])]
    mnb1_acc += [accuracy_score(fold, mnb1_predy[i])]
    
    print('SVM1'+str(i+1), svm1_acc[i])
#     print('SVM2'+str(i+1), svm2_acc[i])
#     print('SVM3'+str(i+1), svm3_acc[i])
#     print('SVM4'+str(i+1), svm4_acc[i])
    print('SVM5'+str(i+1), svm5_acc[i])
    print('SVM6'+str(i+1), svm6_acc[i])
    print('KNN1'+str(i+1), knn1_acc[i])
    print('KNN2'+str(i+1), knn2_acc[i])
    print('RFC1'+str(i+1), rfc1_acc[i])
    print('RFC2'+str(i+1), rfc2_acc[i])
    print('RFC3'+str(i+1), rfc3_acc[i])
    print('MLP1'+str(i+1), mlp1_acc[i])
    print('MLP2'+str(i+1), mlp2_acc[i])
    print('LOGR1'+str(i+1), logr1_acc[i])
    print('LOGR2'+str(i+1), logr2_acc[i])
    print('BNB1'+str(i+1), bnb1_acc[i])
    print('CNB1'+str(i+1), cnb1_acc[i])
    print('GNB1'+str(i+1), gnb1_acc[i])
    print('MNB1'+str(i+1), mnb1_acc[i])


# In[58]:


arr = [svm1_predy, svm2_predy, svm3_predy, svm4_predy, svm5_predy, svm6_predy, knn1_predy, knn2_predy, rfc1_predy, rfc2_predy, rfc3_predy, mlp1_predy, mlp2_predy, logr1_predy, logr2_predy, bnb1_predy, cnb1_predy, gnb1_predy, mnb1_predy]
with open('pickle-files/testing_predy', 'wb') as file:
    pickle.dump(arr, file)


# In[59]:


# Training Accuracy:

svm1_predy_train = []
svm2_predy_train = []
svm3_predy_train = []
svm4_predy_train = []
svm5_predy_train = []
svm6_predy_train = []
knn1_predy_train = []
knn2_predy_train = []
rfc1_predy_train = []
rfc2_predy_train = []
rfc3_predy_train = []
mlp1_predy_train = []
mlp2_predy_train = []
logr1_predy_train = []
logr2_predy_train = []
bnb1_predy_train = []
cnb1_predy_train = []
gnb1_predy_train = []
mnb1_predy_train = []

for i in range(n_folds):
    svm1_predy_train += [svm1[i].predict(df_n[i])]
#     svm2_predy_train += [svm2[i].predict(df_n[i])]
#     svm3_predy_train += [svm3[i].predict(df_n[i])]
#     svm4_predy_train += [svm4[i].predict(df_n[i])]
    svm5_predy_train += [svm5[i].predict(df_n[i])]
    svm6_predy_train += [svm6[i].predict(df_n[i])]
    knn1_predy_train += [knn1[i].predict(df_n[i])]
    knn2_predy_train += [knn2[i].predict(df_n[i])]
    rfc1_predy_train += [rfc1[i].predict(df_n[i])]
    rfc2_predy_train += [rfc2[i].predict(df_n[i])]
    rfc3_predy_train += [rfc3[i].predict(df_n[i])]
    mlp1_predy_train += [mlp1[i].predict(df_n[i])]
    mlp2_predy_train += [mlp2[i].predict(df_n[i])]
    logr1_predy_train += [logr1[i].predict(df_n[i])]
    logr2_predy_train += [logr2[i].predict(df_n[i])]
    bnb1_predy_train += [bnb1[i].predict(df_n[i])]
    cnb1_predy_train += [cnb1[i].predict(df_n[i])]
    gnb1_predy_train += [gnb1[i].predict(df_n[i])]
    mnb1_predy_train += [mnb1[i].predict(df_n[i])]

svm1_acc_train = []
svm2_acc_train = []
svm3_acc_train = []
svm4_acc_train = []
svm5_acc_train = []
svm6_acc_train = []
knn1_acc_train = []
knn2_acc_train = []
rfc1_acc_train = []
rfc2_acc_train = []
rfc3_acc_train = []
mlp1_acc_train = []
mlp2_acc_train = []
logr1_acc_train = []
logr2_acc_train = []
bnb1_acc_train = []
cnb1_acc_train = []
gnb1_acc_train = []
mnb1_acc_train = []

for i in range(n_folds):
    svm1_acc_train += [accuracy_score(df_y_n[i], svm1_predy_train[i])]
#     svm2_acc_train += [accuracy_score(df_y_n[i], svm2_predy_train[i])]
#     svm3_acc_train += [accuracy_score(df_y_n[i], svm3_predy_train[i])]
#     svm4_acc_train += [accuracy_score(df_y_n[i], svm4_predy_train[i])]
    svm5_acc_train += [accuracy_score(df_y_n[i], svm5_predy_train[i])]
    svm6_acc_train += [accuracy_score(df_y_n[i], svm6_predy_train[i])]
    knn1_acc_train += [accuracy_score(df_y_n[i], knn1_predy_train[i])]
    knn2_acc_train += [accuracy_score(df_y_n[i], knn2_predy_train[i])]
    rfc1_acc_train += [accuracy_score(df_y_n[i], rfc1_predy_train[i])]
    rfc2_acc_train += [accuracy_score(df_y_n[i], rfc2_predy_train[i])]
    rfc3_acc_train += [accuracy_score(df_y_n[i], rfc3_predy_train[i])]
    mlp1_acc_train += [accuracy_score(df_y_n[i], mlp1_predy_train[i])]
    mlp2_acc_train += [accuracy_score(df_y_n[i], mlp2_predy_train[i])]
    logr1_acc_train += [accuracy_score(df_y_n[i], logr1_predy_train[i])]
    logr2_acc_train += [accuracy_score(df_y_n[i], logr2_predy_train[i])]
    bnb1_acc_train += [accuracy_score(df_y_n[i], bnb1_predy_train[i])]
    cnb1_acc_train += [accuracy_score(df_y_n[i], cnb1_predy_train[i])]
    gnb1_acc_train += [accuracy_score(df_y_n[i], gnb1_predy_train[i])]
    mnb1_acc_train += [accuracy_score(df_y_n[i], mnb1_predy_train[i])]
    
    
    print('SVM1'+str(i+1), svm1_acc_train[i])
#     print('SVM2'+str(i+1), svm2_acc_train[i])
#     print('SVM3'+str(i+1), svm3_acc_train[i])
#     print('SVM4'+str(i+1), svm4_acc_train[i])
    print('SVM5'+str(i+1), svm5_acc_train[i])
    print('SVM6'+str(i+1), svm6_acc_train[i])
    print('KNN1'+str(i+1), knn1_acc_train[i])
    print('KNN2'+str(i+1), knn2_acc_train[i])
    print('RFC1'+str(i+1), rfc1_acc_train[i])
    print('RFC2'+str(i+1), rfc2_acc_train[i])
    print('RFC3'+str(i+1), rfc3_acc_train[i])
    print('MLP1'+str(i+1), mlp1_acc_train[i])
    print('MLP2'+str(i+1), mlp2_acc_train[i])
    print('LOGR1'+str(i+1), logr1_acc_train[i])
    print('LOGR2'+str(i+1), logr2_acc_train[i])
    print('BNB1'+str(i+1), bnb1_acc_train[i])
    print('CNB1'+str(i+1), cnb1_acc_train[i])
    print('GNB1'+str(i+1), gnb1_acc_train[i])
    print('MNB1'+str(i+1), mnb1_acc_train[i])


# In[60]:


arr = [svm1_predy_train, svm2_predy_train, svm3_predy_train, svm4_predy_train, svm5_predy_train, svm6_predy_train, knn1_predy_train, knn2_predy_train, rfc1_predy_train, rfc2_predy_train, rfc3_predy_train, mlp1_predy_train, mlp2_predy_train, logr1_predy_train, logr2_predy_train, bnb1_predy_train, cnb1_predy_train, gnb1_predy_train, mnb1_predy_train]
with open('pickle-files/training_predy', 'wb') as file:
    pickle.dump(arr, file)


# In[61]:


print("Average Training Accuracy SVM1:", sum(svm1_acc_train)/n_folds)
# print("Average Training Accuracy SVM2:", sum(svm2_acc_train)/n_folds)
# print("Average Training Accuracy SVM3:", sum(svm3_acc_train)/n_folds)
# print("Average Training Accuracy SVM4:", sum(svm4_acc_train)/n_folds)
print("Average Training Accuracy SVM5:", sum(svm5_acc_train)/n_folds)
print("Average Training Accuracy SVM6:", sum(svm6_acc_train)/n_folds)
print("Average Training Accuracy KNN1:", sum(knn1_acc_train)/n_folds)
print("Average Training Accuracy KNN2:", sum(knn2_acc_train)/n_folds)
print("Average Training Accuracy RFC1:", sum(rfc1_acc_train)/n_folds)
print("Average Training Accuracy RFC2:", sum(rfc2_acc_train)/n_folds)
print("Average Training Accuracy RFC3:", sum(rfc3_acc_train)/n_folds)
print("Average Training Accuracy MLP1:", sum(mlp1_acc_train)/n_folds)
print("Average Training Accuracy MLP2:", sum(mlp2_acc_train)/n_folds)
print("Average Training Accuracy LOGR1:", sum(logr1_acc_train)/n_folds)
print("Average Training Accuracy LOGR2:", sum(logr2_acc_train)/n_folds)
print("Average Training Accuracy BNB1:", sum(bnb1_acc_train)/n_folds)
print("Average Training Accuracy CNB1:", sum(cnb1_acc_train)/n_folds)
print("Average Training Accuracy GNB1:", sum(gnb1_acc_train)/n_folds)
print("Average Training Accuracy MNB1:", sum(mnb1_acc_train)/n_folds)

print("Average Testing Accuracy SVM1:", sum(svm1_acc)/n_folds)
# print("Average Testing Accuracy SVM2:", sum(svm2_acc)/n_folds)
# print("Average Testing Accuracy SVM3:", sum(svm3_acc)/n_folds)
# print("Average Testing Accuracy SVM4:", sum(svm4_acc)/n_folds)
print("Average Testing Accuracy SVM5:", sum(svm5_acc)/n_folds)
print("Average Testing Accuracy SVM6:", sum(svm6_acc)/n_folds)
print("Average Testing Accuracy KNN1:", sum(knn1_acc)/n_folds)
print("Average Testing Accuracy KNN2:", sum(knn2_acc)/n_folds)
print("Average Testing Accuracy RFC1:", sum(rfc1_acc)/n_folds)
print("Average Testing Accuracy RFC2:", sum(rfc2_acc)/n_folds)
print("Average Testing Accuracy RFC3:", sum(rfc3_acc)/n_folds)
print("Average Testing Accuracy MLP1:", sum(mlp1_acc)/n_folds)
print("Average Testing Accuracy MLP2:", sum(mlp2_acc)/n_folds)
print("Average Testing Accuracy LOGR1:", sum(logr1_acc)/n_folds)
print("Average Testing Accuracy LOGR2:", sum(logr2_acc)/n_folds)
print("Average Testing Accuracy BNB1:", sum(bnb1_acc)/n_folds)
print("Average Testing Accuracy CNB1:", sum(cnb1_acc)/n_folds)
print("Average Testing Accuracy GNB1:", sum(gnb1_acc)/n_folds)
print("Average Testing Accuracy MNB1:", sum(mnb1_acc)/n_folds)


# In[62]:


# F1 Scores

svm1_f1 = []
svm2_f1 = []
svm3_f1 = []
svm4_f1 = []
svm5_f1 = []
svm6_f1 = []
knn1_f1 = []
knn2_f1 = []
rfc1_f1 = []
rfc2_f1 = []
rfc3_f1 = []
mlp1_f1 = []
mlp2_f1 = []
logr1_f1 = []
logr2_f1 = []
bnb1_f1 = []
cnb1_f1 = []
gnb1_f1 = []
mnb1_f1 = []

for i,fold in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    svm1_f1 += [f1_score(fold, svm1_predy[i])]
#     svm2_f1 += [f1_score(fold, svm2_predy[i])]
#     svm3_f1 += [f1_score(fold, svm3_predy[i])]
#     svm4_f1 += [f1_score(fold, svm4_predy[i])]
    svm5_f1 += [f1_score(fold, svm5_predy[i])]
    svm6_f1 += [f1_score(fold, svm6_predy[i])]
    knn1_f1 += [f1_score(fold, knn1_predy[i])]
    knn2_f1 += [f1_score(fold, knn2_predy[i])]
    rfc1_f1 += [f1_score(fold, rfc1_predy[i])]
    rfc2_f1 += [f1_score(fold, rfc2_predy[i])]
    rfc3_f1 += [f1_score(fold, rfc3_predy[i])]
    mlp1_f1 += [f1_score(fold, mlp1_predy[i])]
    mlp2_f1 += [f1_score(fold, mlp2_predy[i])]
    logr1_f1 += [f1_score(fold, logr1_predy[i])]
    logr2_f1 += [f1_score(fold, logr2_predy[i])]
    bnb1_f1 += [f1_score(fold, bnb1_predy[i])]
    cnb1_f1 += [f1_score(fold, cnb1_predy[i])]
    gnb1_f1 += [f1_score(fold, gnb1_predy[i])]
    mnb1_f1 += [f1_score(fold, mnb1_predy[i])]
    
    print('SVM1'+str(i+1), svm1_f1[i])
#     print('SVM2'+str(i+1), svm2_f1[i])
#     print('SVM3'+str(i+1), svm3_f1[i])
#     print('SVM4'+str(i+1), svm4_f1[i])
    print('SVM5'+str(i+1), svm5_f1[i])
    print('SVM6'+str(i+1), svm6_f1[i])
    print('KNN1'+str(i+1), knn1_f1[i])
    print('KNN2'+str(i+1), knn2_f1[i])
    print('RFC1'+str(i+1), rfc1_f1[i])
    print('RFC2'+str(i+1), rfc2_f1[i])
    print('RFC3'+str(i+1), rfc3_f1[i])
    print('MLP1'+str(i+1), mlp1_f1[i])
    print('MLP2'+str(i+1), mlp1_f1[i])
    print('LOGR1'+str(i+1), logr1_f1[i])
    print('LOGR2'+str(i+1), logr2_f1[i])
    print('BNB1'+str(i+1), bnb1_f1[i])
    print('CNB1'+str(i+1), cnb1_f1[i])
    print('GNB1'+str(i+1), gnb1_f1[i])
    print('MNB1'+str(i+1), mnb1_f1[i])


# In[63]:


print('Average F1-Score SVM1:', sum(svm1_f1)/n_folds)
# print('Average F1-Score SVM2:', sum(svm2_f1)/n_folds)
# print('Average F1-Score SVM3:', sum(svm3_f1)/n_folds)
# print('Average F1-Score SVM4:', sum(svm4_f1)/n_folds)
print('Average F1-Score SVM5:', sum(svm5_f1)/n_folds)
print('Average F1-Score SVM6:', sum(svm6_f1)/n_folds)
print('Average F1-Score KNN1:', sum(knn1_f1)/n_folds)
print('Average F1-Score KNN2:', sum(knn2_f1)/n_folds)
print('Average F1-Score RFC1:', sum(rfc1_f1)/n_folds)
print('Average F1-Score RFC2:', sum(rfc2_f1)/n_folds)
print('Average F1-Score RFC3:', sum(rfc3_f1)/n_folds)
print('Average F1-Score MLP1:', sum(mlp1_f1)/n_folds)
print('Average F1-Score MLP2:', sum(mlp2_f1)/n_folds)
print('Average F1-Score LOGR1:', sum(logr1_f1)/n_folds)
print('Average F1-Score LOGR2:', sum(logr2_f1)/n_folds)
print('Average F1-Score BNB1:', sum(bnb1_f1)/n_folds)
print('Average F1-Score CNB1:', sum(cnb1_f1)/n_folds)
print('Average F1-Score GNB1:', sum(gnb1_f1)/n_folds)
print('Average F1-Score MNB1:', sum(mnb1_f1)/n_folds)


# In[64]:


# estimators = []
# for i in range(n_folds):
# #     estimators += ('psvm', svm1[i])
# #     estimators += ('psvm', svm2[i])
# #     estimators += ('psvm', svm3[i])
# #     estimators += ('psvm', svm4[i])
#     estimators += ('rsvm', svm5[i])
# #     estimators += ('lsvm', svm6[i])
#     estimators += ('knn', knn1[i])
#     estimators += ('knn', knn2[i])
#     estimators += ('rf', rfc1[i])
#     estimators += ('rf', rfc2[i])
#     estimators += ('rf', rfc3[i])
#     estimators += ('mlp', mlp1[i])
#     estimators += ('mlp', mlp2[i])
#     estimators += ('lr', logr1[i])
#     estimators += ('lr', logr2[i])
#     estimators += ('bnb', bnb1[i])
#     estimators += ('cnb', cnb1[i])
#     estimators += ('gnb', gnb1[i])
#     estimators += ('mnb', mnb1[i])
# vclf1 = VotingClassifier(estimators=estimators,  voting='hard')
# vclf2 = VotingClassifier(estimators=estimators,  voting='soft')

# vclf1.fit()

# vclf1_pred = []
# vclf2_pred = []
# vclf1_pred_train = []
# vclf2_pred_train = []

# for i in range(n_folds):
#     vclf1_pred_train += [vclf1.predict(df_n[i])]
#     vclf2_pred_train += [vclf2.predict(df_n[i])]
# for i in [dfa, dfb, dfc, dfd, dfe]:
#     vclf1_pred += [vclf1.predict(i)]
#     vclf2_pred += [vclf2.predict(i)]
    
# for ind, i in zip(range(n_folds), [dfa, dfb, dfc, dfd, dfe]):
#     print('Hard Voting Classifier, Fold ' + str(i+1) + ' :', accuracy_score(fold, vclf1_pred[i]))
#     print('Soft Voting Classifier, Fold ' + str(i+1) + ' :', accuracy_score(fold, vclf2_pred[i]))
# for i in range(n_folds):
#     print('Hard Voting Classifier, Fold ' + str(i+1) + ' :', f1_score(df_yn[i], vclf1_pred_train[i]))
#     print('Soft Voting Classifier, Fold ' + str(i+1) + ' :', f1_score(df_yn[i], vclf2_pred_train[i]))


# In[65]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(svm1_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for poly 1 SVM for fold ' + str(i+1))
    plt.show()


# In[66]:


# for i in range(n_folds):
#     fpr, tpr, _ = roc_curve(svm2_predy_train[i], df_y_n[i], drop_intermediate=False)
#     plt.plot(fpr, tpr, color='red')
#     plt.xlabel('fpr')
#     plt.ylabel('tpr')
#     plt.title('ROC curve for poly 2 SVM for fold ' + str(i+1))
#     plt.show()


# In[67]:


# for i in range(n_folds):
#     fpr, tpr, _ = roc_curve(svm3_predy_train[i], df_y_n[i], drop_intermediate=False)
#     plt.plot(fpr, tpr, color='red')
#     plt.xlabel('fpr')
#     plt.ylabel('tpr')
#     plt.title('ROC curve for poly 3 SVM for fold ' + str(i+1))
#     plt.show()


# In[68]:


# for i in range(n_folds):
#     fpr, tpr, _ = roc_curve(svm4_predy_train[i], df_y_n[i], drop_intermediate=False)
#     plt.plot(fpr, tpr, color='red')
#     plt.xlabel('fpr')
#     plt.ylabel('tpr')
#     plt.title('ROC curve for poly 4 SVM for fold ' + str(i+1))
#     plt.show()


# In[69]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(svm5_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for RBF SVM for fold ' + str(i+1))
    plt.show()


# In[70]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(svm6_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for linear SVM for fold ' + str(i+1))
    plt.show()


# In[71]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(knn1_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for 2 K-Nearest Neighbours for fold ' + str(i+1))
    plt.show()


# In[72]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(knn2_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for 3 K-Nearest Neighbours for fold ' + str(i+1))
    plt.show()


# In[73]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(rfc1_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for 10 RFC for fold ' + str(i+1))
    plt.show()


# In[74]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(rfc2_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for 15 RFC for fold ' + str(i+1))
    plt.show()


# In[75]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(rfc3_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for 1000 RFC for fold ' + str(i+1))
    plt.show()


# In[76]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(mlp1_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for 1000 MLP for fold ' + str(i+1))
    plt.show()


# In[77]:


for i in range(n_folds):    
    fpr, tpr, _ = roc_curve(mlp2_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for 2000 MLP for fold ' + str(i+1))
    plt.show()


# In[78]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(logr1_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for logistic regression (l1) for fold ' + str(i+1))
    plt.show()


# In[79]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(logr2_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for logistic regression (l2) for fold ' + str(i+1))
    plt.show()


# In[80]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(bnb1_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for Bernoulli NB for fold ' + str(i+1))
    plt.show()


# In[81]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(cnb1_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for Complement NB for fold ' + str(i+1))
    plt.show()


# In[82]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(gnb1_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for Gaussian NB for fold ' + str(i+1))
    plt.show()


# In[83]:


for i in range(n_folds):
    fpr, tpr, _ = roc_curve(mnb1_predy_train[i], df_y_n[i], drop_intermediate=False)
    plt.plot(fpr, tpr, color='red')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve for Multinomial NB for fold ' + str(i+1))
    plt.show()


# In[84]:


# Testing Data Confusion Matrices:


# In[85]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm1 = confusion_matrix(y, svm1_predy[i])
    print(cm1)
    cm1df = pd.DataFrame(cm1, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm1df, annot=True)


# In[86]:


# for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
#     fig = plt.figure(i+1)
#     cm2 = confusion_matrix(y, svm2_predy[i])
#     print(cm2)
#     cm2df = pd.DataFrame(cm2, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
#     seaborn.heatmap(cm2df, annot=True)


# In[87]:


# for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
#     fig = plt.figure(i+1)
#     cm3 = confusion_matrix(y, svm3_predy[i])
#     print(cm3)
#     cm3df = pd.DataFrame(cm3, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
#     seaborn.heatmap(cm3df, annot=True)


# In[88]:


# for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
#     fig = plt.figure(i+1)
#     cm4 = confu3sion_matrix(y, svm4_predy[i])
#     print(cm4)
#     cm4df = pd.DataFrame(cm4, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
#     seaborn.heatmap(cm4df, annot=True)


# In[89]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm5 = confusion_matrix(y, svm5_predy[i])
    print(cm5)
    cm5df = pd.DataFrame(cm5, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm5df, annot=True)


# In[90]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm6 = confusion_matrix(y, svm6_predy[i])
    print(cm6)
    cm6df = pd.DataFrame(cm6, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm6df, annot=True)


# In[91]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm7 = confusion_matrix(y, rfc1_predy[i])
    print(cm7)
    cm7df = pd.DataFrame(cm7, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm7df, annot=True)


# In[92]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm8 = confusion_matrix(y, rfc2_predy[i])
    print(cm8)
    cm8df = pd.DataFrame(cm8, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm8df, annot=True)


# In[93]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm9 = confusion_matrix(y, rfc3_predy[i])
    print(cm9)
    cm9df = pd.DataFrame(cm9, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm9df, annot=True)


# In[94]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm10 = confusion_matrix(y, knn1_predy[i])
    print(cm10)
    cm10df = pd.DataFrame(cm10, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm10df, annot=True)


# In[95]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm11 = confusion_matrix(y, knn2_predy[i])
    print(cm11)
    cm11df = pd.DataFrame(cm11, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm11df, annot=True)


# In[96]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm12 = confusion_matrix(y, mlp1_predy[i])
    print(cm12)
    cm12df = pd.DataFrame(cm12, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm12df, annot=True)


# In[97]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm13 = confusion_matrix(y, mlp2_predy[i])
    print(cm13)
    cm13df = pd.DataFrame(cm13, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm13df, annot=True)


# In[98]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm14 = confusion_matrix(y, logr1_predy[i])
    print(cm14)
    cm14df = pd.DataFrame(cm14, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm14df, annot=True)


# In[99]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm15 = confusion_matrix(y, logr2_predy[i])
    print(cm15)
    cm15df = pd.DataFrame(cm15, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm15df, annot=True)


# In[100]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm16 = confusion_matrix(y, bnb1_predy[i])
    print(cm16)
    cm16df = pd.DataFrame(cm16, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm16df, annot=True)


# In[101]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm17 = confusion_matrix(y, cnb1_predy[i])
    print(cm17)
    cm17df = pd.DataFrame(cm17, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm17df, annot=True)


# In[102]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm18 = confusion_matrix(y, gnb1_predy[i])
    print(cm18)
    cm18df = pd.DataFrame(cm18, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm18df, annot=True)


# In[103]:


for i,y in zip(range(n_folds), [df_ya, df_yb, df_yc, df_yd, df_ye]):
    fig = plt.figure(i+1)
    cm19 = confusion_matrix(y, mnb1_predy[i])
    print(cm19)
    cm19df = pd.DataFrame(cm19, index = ["admitted",'not admitted'], columns = ["admitted",'not admitted'])
    seaborn.heatmap(cm19df, annot=True)


# In[104]:


#TensorFlow MLP
import tensorflow as tf

n_nodes_hl1 = 30
n_nodes_hl2 = 45
n_nodes_hl3 = 16

n_classes = 2
batch_size = 100

x = tf.placeholder('float', [None, 15])
y = tf.placeholder('float',[None,2])

def neuralnetwork(data):
    hiddenlayer1 = {'weights':tf.Variable(tf.random_normal([15, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hiddenlayer2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    # hiddenlayer3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    outputlayer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

    l1 = tf.add(tf.matmul(data,hiddenlayer1['weights']), hiddenlayer1['biases'])
    l1 = tf.nn.tanh(l1)
    l2 = tf.add(tf.matmul(l1,hiddenlayer2['weights']), hiddenlayer2['biases'])
    l2 = tf.nn.tanh(l2)
    
    # l3 = tf.add(tf.matmul(l2,hiddenlayer3['weights']), hiddenlayer3['biases'])
    # l3 = tf.nn.tanh(l3)
    output = tf.matmul(l2,outputlayer['weights']) + outputlayer['biases']
    output=tf.nn.softmax(output)
    return output


# In[105]:


cwd = os.getcwd()
print(cwd)


# In[106]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
df_yaten = enc.fit_transform(df_ya.reshape(-1,1)).toarray()
df_ybcdeten = enc.fit_transform(df_ybcde.reshape(-1,1)).toarray()
df_ybten = enc.fit_transform(df_yb.reshape(-1,1)).toarray()
df_yacdeten = enc.fit_transform(df_yacde.reshape(-1,1)).toarray()
df_ycten = enc.fit_transform(df_yc.reshape(-1,1)).toarray()
df_yabdeten = enc.fit_transform(df_yabde.reshape(-1,1)).toarray()
df_ydten = enc.fit_transform(df_yd.reshape(-1,1)).toarray()
df_yabceten = enc.fit_transform(df_yabce.reshape(-1,1)).toarray()
df_yeten = enc.fit_transform(df_ye.reshape(-1,1)).toarray()
df_yabcdten = enc.fit_transform(df_yabcd.reshape(-1,1)).toarray()


# In[107]:


def train_neural_network(x):
    prediction = neuralnetwork(x)
    eploss=[]
    eplosstrain=[]
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    # print(optimizer)
    
    hm_epochs = 1000
    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            epoch_losstrain=0
            # for i in range(40613):
            #     epoch_x, epoch_y =dfbcde[i].reshape(1,-1),df_ybcde[i].reshape(1,-1) 
            #     # print(type((epoch_x)))
            #     lam, c = sess.run([optimizer, cost], feed_dict={x: (epoch_x), y: (epoch_y)})
            #     epoch_loss += c
            #     print(lam,c)
            # for index in range(0,38868,10000):
            #   epoch_x, epoch_y =dfbcde[index:index+10000,:],df_ybcdeten[index:index+10000,:]
            #   lam, c = sess.run([optimizer, cost], feed_dict={x: (epoch_x), y: (epoch_y)})
            #   # lam, c1 = sess.run([optimizer, cost], feed_dict={x: (dfa), y: (df_yaten)})
            epoch_x, epoch_y =dfbcde,df_ybcdeten
            lam, c = sess.run([optimizer, cost], feed_dict={x: (epoch_x), y: (epoch_y)})
            epoch_loss += c
            eploss.append(epoch_loss)

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        print(prediction)
        predy=prediction.eval({x:dfa,y:df_yaten})

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("accuracy on train:",accuracy.eval({x:dfbcde,y:df_ybcdeten}))
        print('Accuracy on test :',accuracy.eval({x:dfa, y:df_yaten}))
    return eploss,predy


# In[108]:


losslist, predy = train_neural_network(x)


# In[109]:


predy=np.argmax(predy,axis=1)
cm8 = confusion_matrix(df_ya,predy)
plt.figure(figsize = (10,7))
cm8df = pd.DataFrame(cm8, index = ["1",'0'], columns = ["1",'0'])
seaborn.set(font_scale=1.4)#for label size
seaborn.heatmap(cm8df, annot=True,annot_kws={"size": 16})# font size


# In[110]:


fpr, tpr, _ = roc_curve(predy, df_ya, drop_intermediate=False)
plt.plot(fpr, tpr, color='red')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve fr Tensorflow 1')
plt.show()


# In[111]:


seq = np.arange(1000).tolist()
plt.plot(seq, losslist)
plt.show()


# In[112]:


#TensorFlow MLP

n_nodes_hl1 =30
n_nodes_hl2=45
n_nodes_hl3=10
n_nodes_hl4=8
n_classes = 2

x = tf.placeholder('float', [None, 15])
y = tf.placeholder('float',[None,2])

def neuralnetwork2(data):
    hiddenlayer1 = {'weights':tf.Variable(tf.random_normal([15, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hiddenlayer2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
#     hiddenlayer3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    # hiddenlayer4 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
    outputlayer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes])),}

    l1 = tf.add(tf.matmul(data,hiddenlayer1['weights']), hiddenlayer1['biases'])
    l1 = tf.nn.tanh(l1)
    l2 = tf.add(tf.matmul(l1,hiddenlayer2['weights']), hiddenlayer2['biases'])
    l2 = tf.nn.tanh(l2)
#     l3 = tf.add(tf.matmul(l2,hiddenlayer3['weights']), hiddenlayer3['biases'])
#     l3 = tf.nn.tanh(l3)
    # l4 = tf.add(tf.matmul(l3,hiddenlayer4['weights']), hiddenlayer4['biases'])
    # l4 = tf.nn.tanh(l4)

    output = tf.matmul(l2,outputlayer['weights']) + outputlayer['biases']
    output=tf.nn.softmax(output)

    return output


# In[113]:


def train_neural_network2(x, dfx, dfy, dftestx, dftesty):
    prediction = neuralnetwork2(x)
    eploss=[]
    eplosstrain=[]
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y ))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    # print(optimizer)
    
    epochs = 1000
    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_losstrain=0
            # for i in range(40613):
            #     epoch_x, epoch_y =dfbcde[i].reshape(1,-1),df_ybcde[i].reshape(1,-1) 
            #     # print(type((epoch_x)))
            #     lam, c = sess.run([optimizer, cost], feed_dict={x: (epoch_x), y: (epoch_y)})
            #     epoch_loss += c
            #     print(lam,c)
            for index in range(0,38867,10000):
                epoch_x, epoch_y =dfx[index:index+10000,:],dfy[index:index+10000,:]
                lam, c = sess.run([optimizer, cost], feed_dict={x: (epoch_x), y: (epoch_y)})
                # lam, c1 = sess.run([optimizer, cost], feed_dict={x: (dfa), y: (df_yaten)})
                epoch_loss += c
            eploss.append(epoch_loss)
  
            print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)
        print(prediction)
        # confusion = tf.confusion_matrix(labels=y, predictions=prediction, num_classes=2)
        predy=prediction.eval({x:dftestx,y:dftesty})
        print(predy)
        print(np.array_equal(predy,dftesty))
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("accuracy on train:",accuracy.eval({x:dfx,y:dfy}))
        print('Accuracy on test :',accuracy.eval({x:dftestx, y:dftesty}))
        # print(confusion.eval(session=sess))
        print(dftesty)
    return eploss, predy


# In[114]:


losslist2, predy2 = train_neural_network2(x, dfacde, df_yacdeten, dfb, df_ybten)
predy2 = np.argmax(predy2, axis=1)
print(predy2)


# In[115]:


cm8 = confusion_matrix(df_yb,predy2)
plt.figure(figsize = (10,7))
cm8df = pd.DataFrame(cm8, index = ["1",'0'], columns = ["1",'0'])
seaborn.set(font_scale=1.4)#for label size
seaborn.heatmap(cm8df, annot=True,annot_kws={"size": 16})# font size

# plt.show()
# ig, ax = plt.subplots()
# cm = confusion_matrix(df_yb, predy2)

# im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# ax.figure.colorbar(im, ax=ax)

# ax.set(yticks=[-0.5, 1.5], 
#        xticks=[0, 1], 
#        yticklabels=[1,0], 
#        xticklabels=[1,0])
# # ax.yaxis.set_major_locator(ticker.IndexLocater(base=1, offset=0.5))
# ax.yaxis.set_major_locator(mat.ticker.IndexLocator(base=1, offset=0.5))
# seaborn.heatmap(cm8df, annot=True)


# In[116]:


fpr, tpr, _ = roc_curve(predy2, df_yb, drop_intermediate=False)
plt.plot(fpr, tpr, color='red')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve fr Tensorflow 2')
plt.show()


# In[117]:


losslist3,predy3 = train_neural_network2(x, dfabde, df_yabdeten, dfc, df_ycten)
predy3=np.argmax(predy3,axis=1)


# In[118]:


cm8 = confusion_matrix(df_yc,predy3)
plt.figure(figsize = (10,7))
cm8df = pd.DataFrame(cm8, index = ["1",'0'], columns = ["1",'0'])
seaborn.set(font_scale=1.4)#for label size
seaborn.heatmap(cm8df, annot=True,annot_kws={"size": 16})# font size


# In[119]:


fpr, tpr, _ = roc_curve(predy3, df_yc, drop_intermediate=False)
plt.plot(fpr, tpr, color='red')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve fr Tensorflow 3')
plt.show()


# In[120]:


losslist4,predy4 = train_neural_network2(x, dfabce, df_yabceten, dfd, df_ydten)
predy4=np.argmax(predy4,axis=1)


# In[121]:


cm8 = confusion_matrix(df_yd,predy4)
plt.figure(figsize = (10,7))
cm8df = pd.DataFrame(cm8, index = ["1",'0'], columns = ["1",'0'])
seaborn.set(font_scale=1.4)#for label size
seaborn.heatmap(cm8df, annot=True,annot_kws={"size": 16})# font size


# In[122]:


fpr, tpr, _ = roc_curve(predy4, df_yd, drop_intermediate=False)
plt.plot(fpr, tpr, color='red')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve fr Tensorflow 4')
plt.show()


# In[123]:


losslist5,predy5 = train_neural_network2(x, dfabcd, df_yabcdten, dfe, df_yeten)
predy5 = np.argmax(predy5,axis=1)


# In[124]:


cm8 = confusion_matrix(df_ye,predy5)
plt.figure(figsize = (10,7))
cm8df = pd.DataFrame(cm8, index = ["1",'0'], columns = ["1",'0'])
seaborn.set(font_scale=1.4)#for label size
seaborn.heatmap(cm8df, annot=True,annot_kws={"size": 16})# font size


# In[125]:


fpr, tpr, _ = roc_curve(predy5, df_ye, drop_intermediate=False)
plt.plot(fpr, tpr, color='red')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve fr Tensorflow 5')
plt.show()


# In[126]:


seq = np.arange(1000).tolist()
plt.plot(seq, losslist2)
plt.show()


# In[127]:


plt.plot(seq, losslist3)
plt.show()


# In[128]:


plt.plot(seq, losslist4)
plt.show()


# In[129]:


plt.plot(seq, losslist5)
plt.show()


# In[130]:


print("F1 Score for fold 1 of Tensorflow model:", f1_score(df_ya, predy))
print("F1 Score for fold 2 of Tensorflow model:", f1_score(df_yb, predy2))
print("F1 Score for fold 3 of Tensorflow model:", f1_score(df_yc, predy3))
print("F1 Score for fold 4 of Tensorflow model:", f1_score(df_yd, predy4))
print("F1 Score for fold 5 of Tensorflow model:", f1_score(df_ye, predy5))


# In[147]:


from sklearn.metrics import classification_report, recall_score, precision_score, precision_recall_curve


# In[146]:


# from inspect import signature
# # https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
# print(len(df_ya), len(svm1_predy[0]))
# precision, recall, _ = precision_recall_curve(df_ya, svm1_predy[0])
# print(precision, recall, _)
# # step_kwargs = ({'step': 'post'}
# #                if 'step' in signature(plt.fill_between).parameters
# #                else {})
# plt.step(recall, precision, color='red', alpha=0.2, where='post')
# # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


# In[151]:


print(classification_report(df_ya, svm1_predy[0]))
print(classification_report(df_yb, svm1_predy[1]))
print(classification_report(df_yc, svm1_predy[2]))
print(classification_report(df_yd, svm1_predy[3]))
print(classification_report(df_ye, svm1_predy[4]))

print(classification_report(df_ya, svm5_predy[0]))
print(classification_report(df_yb, svm5_predy[1]))
print(classification_report(df_yc, svm5_predy[2]))
print(classification_report(df_yd, svm5_predy[3]))
print(classification_report(df_ye, svm5_predy[4]))

print(classification_report(df_ya, svm6_predy[0]))
print(classification_report(df_yb, svm6_predy[1]))
print(classification_report(df_yc, svm6_predy[2]))
print(classification_report(df_yd, svm6_predy[3]))
print(classification_report(df_ye, svm6_predy[4]))

print(classification_report(df_ya, knn1_predy[0]))
print(classification_report(df_yb, knn1_predy[1]))
print(classification_report(df_yc, knn1_predy[2]))
print(classification_report(df_yd, knn1_predy[3]))
print(classification_report(df_ye, knn1_predy[4]))

print(classification_report(df_ya, knn2_predy[0]))
print(classification_report(df_yb, knn2_predy[1]))
print(classification_report(df_yc, knn2_predy[2]))
print(classification_report(df_yd, knn2_predy[3]))
print(classification_report(df_ye, knn2_predy[4]))

print(classification_report(df_ya, rfc1_predy[0]))
print(classification_report(df_yb, rfc1_predy[1]))
print(classification_report(df_yc, rfc1_predy[2]))
print(classification_report(df_yd, rfc1_predy[3]))
print(classification_report(df_ye, rfc1_predy[4]))

print(classification_report(df_ya, rfc2_predy[0]))
print(classification_report(df_yb, rfc2_predy[1]))
print(classification_report(df_yc, rfc2_predy[2]))
print(classification_report(df_yd, rfc2_predy[3]))
print(classification_report(df_ye, rfc2_predy[4]))

print(classification_report(df_ya, rfc3_predy[0]))
print(classification_report(df_yb, rfc3_predy[1]))
print(classification_report(df_yc, rfc3_predy[2]))
print(classification_report(df_yd, rfc3_predy[3]))
print(classification_report(df_ye, rfc3_predy[4]))

print(classification_report(df_ya, mlp1_predy[0]))
print(classification_report(df_yb, mlp1_predy[1]))
print(classification_report(df_yc, mlp1_predy[2]))
print(classification_report(df_yd, mlp1_predy[3]))
print(classification_report(df_ye, mlp1_predy[4]))

print(classification_report(df_ya, mlp2_predy[0]))
print(classification_report(df_yb, mlp2_predy[1]))
print(classification_report(df_yc, mlp2_predy[2]))
print(classification_report(df_yd, mlp2_predy[3]))
print(classification_report(df_ye, mlp2_predy[4]))

print(classification_report(df_ya, logr1_predy[0]))
print(classification_report(df_yb, logr1_predy[1]))
print(classification_report(df_yc, logr1_predy[2]))
print(classification_report(df_yd, logr1_predy[3]))
print(classification_report(df_ye, logr1_predy[4]))

print(classification_report(df_ya, logr2_predy[0]))
print(classification_report(df_yb, logr2_predy[1]))
print(classification_report(df_yc, logr2_predy[2]))
print(classification_report(df_yd, logr2_predy[3]))
print(classification_report(df_ye, logr2_predy[4]))

print(classification_report(df_ya, bnb1_predy[0]))
print(classification_report(df_yb, bnb1_predy[1]))
print(classification_report(df_yc, bnb1_predy[2]))
print(classification_report(df_yd, bnb1_predy[3]))
print(classification_report(df_ye, bnb1_predy[4]))

print(classification_report(df_ya, cnb1_predy[0]))
print(classification_report(df_yb, cnb1_predy[1]))
print(classification_report(df_yc, cnb1_predy[2]))
print(classification_report(df_yd, cnb1_predy[3]))
print(classification_report(df_ye, cnb1_predy[4]))

print(classification_report(df_ya, gnb1_predy[0]))
print(classification_report(df_yb, gnb1_predy[1]))
print(classification_report(df_yc, gnb1_predy[2]))
print(classification_report(df_yd, gnb1_predy[3]))
print(classification_report(df_ye, gnb1_predy[4]))

print(classification_report(df_ya, mnb1_predy[0]))
print(classification_report(df_yb, mnb1_predy[1]))
print(classification_report(df_yc, mnb1_predy[2]))
print(classification_report(df_yd, mnb1_predy[3]))
print(classification_report(df_ye, mnb1_predy[4]))


# In[ ]:




