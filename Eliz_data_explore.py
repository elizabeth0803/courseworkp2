import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn 

#data explore

# Load the dataset
train_df = pd.read_csv('train.csv')

from sklearn.model_selection import train_test_split

#pd.set_option('display.max_columns', None)
#display(train_df)


train_df, valid_df = train_test_split(
    train_df,
    test_size = 0.2,
    random_state = 5059
)
train_labels = train_df['Status']
valid_labels = valid_df['Status']

#All categorical attributes have just Y and N which could be binary except
# 'Edema' which has Y, N and S. And 'Drug' has two as well

df_d = train_df[train_df['Status'] == 'D'] 
df_c = train_df[train_df['Status'] =='C']
df_cl = train_df[train_df['Status'] =='CL']


numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train_df.select_dtypes(include=['object']).columns

for column in numerical_cols:
    print(f'{column} D:', np.mean(df_d[column]), 'C:', np.mean(df_c[column]) ,'CL:', np.mean(df_cl[column]))
   
print("The largest difference in mean is in 'N_Days', 'Bilirubin', 'Copper' and 'Alk_Phos'")
numerical_train = train_df[numerical_cols]
numerical_d = df_d[numerical_cols]
numerical_c = df_c[numerical_cols]
numerical_cl = df_cl[numerical_cols]

np_train = numerical_train.to_numpy()
np_d = numerical_d.to_numpy()
np_c = numerical_c.to_numpy()
np_cl = numerical_cl.to_numpy()

for i in range(len(numerical_cols)):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize =(10,5))
    a = np_d[:, i]
    b = np_train[:, i]
    c = np_c[:, i]
    d = np_cl[:, i]

    mi = np.min(b)
    ma = np.max(b)
    ra = ma - mi
    increment = (ma-mi)/15
    
    la = []
    lb = []
    lc = []
    ld = []
    lbc = []
    lbd = []
    lba = []
    for m in range(15):
        la.append(0)
        lb.append(0)
        lc.append(0)
        ld.append(0)
        lba.append(0)
        lbc.append(0)
        lbd.append(0)
        
    for j in range(15):
        low = mi + j*increment
        high = mi + (j+1)*increment
        na = 0
        nb = 0
        nc = 0
        nd = 0
        for k in range(len(b)):
            if b[k] > low and b[k] <= high:
                nb = nb+1
        for l in range(len(a)):
            if a[l] > low and a[l] <= high:
                na = na+1
        for o in range(len(c)):
            if c[o] > low and c[o] <= high:
                nc = nc +1
        for p in range(len(d)):
            if d[p] > low and d[p] <= high:
                nd = nd+1
        lb[j] = nb
        la[j] = na
        lc[j] = nc
        ld[j] = nd
        
        
    lb = np.array(lb)
    la = np.array(la)
    lc = np.array(lc)
    ld = np.array(ld)
    
    for n in range(15):
        if lb[n] == 0:
            lba[n] = 0
            lbc[n] = 0
            lbd[n] = 0
        else:
            lba[n] = la[n]/lb[n]
            lbc[n] = lc[n]/lb[n]
            lbd[n] = ld[n]/lb[n]
    ax1.hist(b, bins = 15, alpha = 0.5, label = " total")
    ax2.hist(b, bins = 15, alpha = 0.5, label = " total")
    ax3.hist(b, bins = 15, alpha = 0.5, label = " total")
    ax1.hist(a, bins = 15, alpha = 0.5, label = "D")
    ax2.hist(c, bins = 15, alpha = 0.5, label = "C")
    ax3.hist(d, bins = 15, alpha = 0.5, label = "CL")
    ax4.plot(lba, label = '%D')
    ax4.plot(lbc, label = '%C')
    ax4.plot(lbd, label = '%CL')
    plt.title(f"{numerical_cols[i]}")
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

print("We should use the columns where the % line shapes change drastically between categories.")

pd.plotting.scatter_matrix(train_df, figsize = (30,30))
plt.show()

print("From this we can see that there are no obvious direct correlations between features.")