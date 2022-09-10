# First we import our libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import our dataset

from sklearn.datasets import load_wine
wine = load_wine()

# Get data keys

print(wine.keys())
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names'])

# In order to understand this dataset better we will check the description
wine.DESCR
  The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators. 
  There are thirteen different measurements taken for different constituents found in the three types of wine
  
# So, from the description we can see that this data measured the differences between the three types of wine.

# Now that we saw the data in detail, we convert to pandas DataFrame format
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

# Check all dataframe feature names

print(df.keys())
Index(['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
       'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 
       'proanthocyanins', 'color_intensity', 'hue',
       'od280/od315_of_diluted_wines', 'proline'],
      dtype='object')

# Show the first 5 rows of dataframe

print(df.head())
   alcohol  ...  proline
0    14.23  ...   1065.0       
1    13.20  ...   1050.0       
2    13.16  ...   1185.0       
3    14.37  ...   1480.0       
4    13.24  ...    735.0   

# We want to get the whole picture, so we expand the df.head to show all columns

pd.set_option('display.max_columns', None)
print(df.head())
alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  \
0    14.23        1.71  2.43               15.6      127.0           2.80   
1    13.20        1.78  2.14               11.2      100.0           2.65   
2    13.16        2.36  2.67               18.6      101.0           2.80   
3    14.37        1.95  2.50               16.8      113.0           3.85   
4    13.24        2.59  2.87               21.0      118.0           2.80   

   flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  \
0        3.06                  0.28             2.29             5.64  1.04   
1        2.76                  0.26             1.28             4.38  1.05   
2        3.24                  0.30             2.81             5.68  1.03   
3        3.49                  0.24             2.18             7.80  0.86   
4        2.69                  0.39             1.82             4.32  1.04   

   od280/od315_of_diluted_wines  proline  
0                          3.92   1065.0  
1                          3.40   1050.0  
2                          3.17   1185.0  
3                          3.45   1480.0  
4                          2.93    735.0

# Checking the shape of the entire dataset

print(df.shape)
(178, 13)

# This shows that this dataset has 178 observations and 13 columns

# Since the values of the dataset are not equally scaled we need to apply z-score standardization to get all features into the same scale. 
# For this, we use Scikit-learn StandardScaler() class which is in the preprocessing submodule in Scikit-learn.

# Import Class
from sklearn.preprocessing import StandardScaler

# Create Object and store in variable Scaler
scaler = StandardScaler()

# Mean and Standard Deviation
scaler.fit(df)

# Transform values and store into df_scaled
df_scaled = scaler.transform(df)

# Now we are ready to apply PCA to our dataset.
from sklearn.decomposition import PCA

# Now we apply PCA with the original number of dimensions to see how well PCA captures the variance of the data
PCA_13 = PCA(n_components=13)
PCA_13.fit(df_scaled)
DF_PCA_13 = PCA_13.transform(df_scaled)

# Since we have set n_components = 13 which is the original number of dimensions in our dataset
# The % variance explained by 13 components should be 100%
sum(PCA_13.explained_variance_ratio_*100)
100.0

# The explained_variance_ratio_ attribute of PCA returns an array
# Which has the values of the percentage of variance explained by each of the components

PCA_13.explained_variance_ratio_*100
array([36.1988481 , 19.20749026, 11.12363054,  7.06903018,  6.56329368,
        4.93582332,  4.23867932,  2.68074895,  2.2221534 ,  1.93001909,
        1.73683569,  1.29823258,  0.79521489])

# In the array above, we can see that there are 13 components.
# The first variable alone captures 36.19% of the variability in the dataset
# And the second variable captures 19.21% of the variability in the dataset by itself and so on.

# If we get the cumulative sum of the array we can see the following
np.cumsum(PCA_13.explained_variance_ratio_ * 100)
array([ 36.1988481 ,  55.40633836,  66.52996889,  73.59899908,
        80.16229276,  85.09811607,  89.3367954 ,  92.01754435,
        94.23969775,  96.16971684,  97.90655253,  99.20478511,
       100.        ])

# Then we create the following plot for better visualization
plt.plot(np.cumsum(PCA_13.explained_variance_ratio_ * 100))
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.savefig('elbow_plot.png', dpi=100)

#### Plot is in Another file ####

# From looking at specific components we can see the following
print("Variance explained by the First principal component =", np.cumsum(PCA_13.explained_variance_ratio_ * 100)[0])
print("Variance explained by the First two principal components =", np.cumsum(PCA_13.explained_variance_ratio_ * 100)[1])
print("Variance explained by the First four principal components =", np.cumsum(PCA_13.explained_variance_ratio_ * 100)[3])
print("Variance explained by the First six principal components =", np.cumsum(PCA_13.explained_variance_ratio_ * 100)[5])
print("Variance explained by the First eight principal components =", np.cumsum(PCA_13.explained_variance_ratio_ * 100)[7])
print("Variance explained by the First ten principal components =", np.cumsum(PCA_13.explained_variance_ratio_ * 100)[9])
Variance explained by the First principal component = 36.19884809992633
Variance explained by the First two principal components = 55.406338356935294
Variance explained by the First four principal components = 73.59899907589929
Variance explained by the First six principal components = 85.09811607477045
Variance explained by the First eight principal components = 92.01754434577262
Variance explained by the First ten principal components = 96.16971684450642

# You can see that the first six components keep about 85% of the variability 
# While reducing 7 (54%) features in the dataset. That is good since the 
# Remaining 7 features only contain around 15% of the variability in the data.

# Now we apply PCA to our dataset with n_components=2. 
# This will project the data into a two-dimensional subspace
# And return 2 components that capture 55.41% of the variability in data
PCA_2 = PCA(n_components=2)
PCA_2.fit(df_scaled)
DF_PCA_2 = PCA_2.transform(df_scaled)

# Here we check the new shape of the dataset
DF_PCA_2.shape
(178, 2)
# As you can see, the dimensionality reduced from 13 to 2

# Here we confirm the three object classes that the data was evaluating
wine.target_names
array(['class_0', 'class_1', 'class_2'], dtype='<U7')
# These classes are the three different type of wine that
# were mentioned at the beginning of this project in the description of the dataset.

# Now we create a 2d scatterplot of the data using the two principal components
plt.figure(figsize=(12,8))
ax = sns.scatterplot(x=DF_PCA_2[:, 0],y=DF_PCA_2[:, 1], s=70,
        hue=wine.target, palette=['orange', 'blue', 'red'])
legend_labels, _= ax.get_legend_handles_labels()
ax.legend(legend_labels, ['Class 0', 'Class 1', 'Class 2'])
plt.title("2D Scatterplot: 55.41% of the variability captured", pad=15)
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.savefig("2d_scatterplot.png")

#### Plot is in 2d_scatterplot graph ####

# By using the two components we can easily separate the three classes
