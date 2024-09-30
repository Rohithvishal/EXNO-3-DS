## EXNO-3:To read the given data and perform Feature Encoding and Transformation process and save the data to a file.
## Name: Rohith J
## Register Number: 24000942
## Department: Artificial Intelligence And Data Science.
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
# 1.Feature Encoding
## Ordinal Encoder:
```
import pandas as pd
df=pd.read_csv('Encoding Data.csv')
df
```
![image](https://github.com/user-attachments/assets/24bf4b3c-dbb6-4980-933a-39ebb096da5b)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
#ordinal encoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[['ord_2']])
```
![image](https://github.com/user-attachments/assets/1cfbff5d-dbf9-4772-97c2-d4189ff58a34)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/c2b7329a-aaba-487b-bed0-49116a365e59)

## Label Encoder:
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/862e5e7d-4552-4111-9595-d2a4fddf926f)

## One Hot Encoder:
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/48d42dea-4e3e-41f1-93bb-685e475c6c05)

## Category Encoder:
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/2012e498-0aa4-4a4e-9841-8cb64005d1ca)

## Binary Encoder:
```
from category_encoders import BinaryEncoder
df=pd.read_csv('data.csv')
df
```
![image](https://github.com/user-attachments/assets/1c34b1c2-537e-456d-a6ab-a1980042c94b)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/463159b7-1ca4-4ff8-9752-614a5cbefd5c)

## Target Encoder:
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/6c8f6c43-8787-4ad4-8cff-ad6ccfcc0e5d)

# 2.0 FUNCTION TRANSFORMATION
## Log Transformation
```
from scipy import stats
import numpy as np
df=pd.read_csv('Data_to_Transform.csv')
df
```
![image](https://github.com/user-attachments/assets/c6f2fd22-201b-45f7-a163-e7fd8a821b57)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/fa89f8a4-a775-4ee5-8a45-338aa0cb91dc)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/9e72c0bd-02bb-4a57-b5d3-b39dc55e26fa)

## Reciprocal Transformation
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/67ad7ca8-ff01-4839-999a-4c749dd5247a)

## Square Root Transformation
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/1585c2f5-642c-4cc3-9b1a-8a851f51196b)

## Square Transformation
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/288486aa-97fb-478a-b7be-9788de7182eb)

# 2.1 POWER TRANSFORMATION
## Boxcox method
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/f0770dfc-e8ce-4079-82c8-394d3f91fb6f)

## Yeojohnson method
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/2a955b55-9360-4b39-b947-a10e08e0339a)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/72e52797-775a-449e-a5c9-37f0c48be233)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/2368d576-78d0-4b76-9695-a53aaf8c4050)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/8c6e2883-c21b-4d3e-aa15-1f69b4e196d1)

# 2.2 Quantile Transformer:
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/71ea27ec-aaf3-475c-aaa8-266bbbe3b3b1)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/3df214f2-193c-44cf-bf1e-b014fe50c14a)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/0f7f8e67-8708-418e-9397-6dba9a642e90)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/165159ac-fb5e-4561-bc35-90f77b57ebde)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/8f58614d-ef3c-48e4-a056-3e2028c24dfd)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/92101de2-e721-4fac-ad5c-60422fe38316)

# RESULT:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file is executed successfully.

       
