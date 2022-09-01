# Principal Component Analysis using Wine data from SKlearn

Firstly, this project was inspired by Stephen Tran (https://www.linkedin.com/in/stephentran96/) and this article by Rukshan
https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0

WHAT WE WILL DO?

This project will look to apply principal component analysis techniques to the wine data from SKLearn.

WHY DO WE DO THIS?

Principal component analysis is one of the ways to reduce dimensionality.
Dimensionality is the number of dimensions (feature names or inputs) that are in a dataset.
Reducing the dimensionality projects data with a high amount of dimensionality to a low amount of dimensionality while retaining as much variation as we can.
While there are some Linear and Non-linear methods to reducing dimensionality, we will be using PCA.
The reason for why we want to reduce dimensionality is because algorithms
are unable to efficiently train on certain data because of the sheer size of the feature space.
Once the data is in a low-dimensional space, algorithms are able to identify interesting patterns more quickly and accurately.

WHY PCA?

PCA considers the correlation among features. If the correlation is very high among a portion of the features, 
PCA will attempt to combine the features with a high amount of correlation and represent this data with a smaller amount of linearly uncorrelated features. 
The algorithm continues this correlation reduction, finding the directions of maximum variance in the 
original high-dimensional data and projecting them onto a smaller dimensional space. 
These newly derived components are known as principal components which will improve the performance of any algorithms.
