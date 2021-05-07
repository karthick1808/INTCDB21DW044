#!/usr/bin/env python
# coding: utf-8

# ## 3.1 Finding an Observation’s Nearest Neighbors

# You need to find an observation’s k nearest observations (neighbors).

# Use scikit-learn’s NearestNeighbors:

# In[2]:


# Load libraries
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# In[3]:


# Load data
iris = datasets.load_iris()
features = iris.data


# In[4]:


# Create standardizer
standardizer = StandardScaler()


# In[5]:


# Standardize features
features_standardized = standardizer.fit_transform(features)


# In[6]:


# Two nearest neighbors
nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)


# In[8]:


# Create an observation
new_observation = [ 1, 1, 1, 1]


# In[9]:


# Find distances and indices of the observation's nearest neighbors
distances, indices = nearest_neighbors.kneighbors([new_observation])


# In[10]:


# View the nearest neighbors
features_standardized[indices]


# We can set the distance metric using the metric parameter:

# In[11]:


# Find two nearest neighbors based on euclidean distance
nearestneighbors_euclidean = NearestNeighbors(
n_neighbors=2, metric='euclidean').fit(features_standardized)


# The distance variable we created contains the actual distance measurement to each
# of the two nearest neighbors:

# In[12]:


# View distances
distances


# In addition, we can use kneighbors_graph to create a matrix indicating each observation’s
# nearest neighbors:

# In[13]:


# Find each observation's three nearest neighbors
# based on euclidean distance (including itself)
nearestneighbors_euclidean = NearestNeighbors(
n_neighbors=3, metric="euclidean").fit(features_standardized)
# List of lists indicating each observation's 3 nearest neighbors
# (including itself)
nearest_neighbors_with_self = nearestneighbors_euclidean.kneighbors_graph(
features_standardized).toarray()
# Remove 1's marking an observation is a nearest neighbor to itself
for i, x in enumerate(nearest_neighbors_with_self):
    x[i] = 0
# View first observation's two nearest neighbors
nearest_neighbors_with_self[0]


# ## Explanation

# In[ ]:










# ## 3.2 Creating a K-Nearest Neighbor Classifier

# Given an observation of unknown class, you need to predict its class based on the
# class of its neighbors

# If the dataset is not very large, use KNeighborsClassifier

# In[16]:


# Load libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


# In[17]:


# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target


# In[18]:


# Create standardizer
standardizer = StandardScaler()


# In[19]:


# Standardize features
X_std = standardizer.fit_transform(X)


# In[20]:


# Train a KNN classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_std, y)


# In[21]:


# Create two observations
new_observations = [[ 0.75, 0.75, 0.75, 0.75],
                        [ 1, 1, 1, 1]]


# In[22]:


# Predict the class of two observations
knn.predict(new_observations)


# In[ ]:


y


# In scikit-learn we can see
# these probabilities using predict_proba

# In[23]:


# View probability each observation is one of three classes
knn.predict_proba(new_observations)


# The class with the highest probability becomes the predicted class. For example, in
# the preceding output, the first observation should be class 1 (Pr = 0.6) while the second
# observation should be class 2 (Pr = 1), and this is just what we see:

# In[ ]:


knn.predict(new_observations)


# ## 3.3 Identifying the Best Neighborhood Size

# You want to select the best value for k in a k-nearest neighbors classifier.

# Use model selection techniques like GridSearchCV

# In[24]:


# Load libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV


# In[25]:


# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target


# In[26]:


# Create standardizer
standardizer = StandardScaler()


# In[27]:


# Standardize features
features_standardized = standardizer.fit_transform(features)


# In[28]:


# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


# In[29]:


# Create a pipeline
pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])


# In[30]:


pipe


# In[31]:


# Create space of candidate values
search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]


# In[33]:


# Create grid search
classifier = GridSearchCV(
pipe, search_space, cv=5, verbose=0).fit(features_standardized, target)


# When that is completed, we can see the k that produces the
# best model

# In[34]:


#Best neighborhood size (k)
classifier.best_estimator_.get_params()["knn__n_neighbors"]


# ## 3.4 Creating a Radius-Based Nearest Neighbor Classifier

# Given an observation of unknown class, you need to predict its class based on the
# class of all observations within a certain distance.

# Use RadiusNeighborsClassifier:

# In[35]:


# Load libraries
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets


# In[36]:


# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target


# In[37]:


# Create standardizer
standardizer = StandardScaler()


# In[38]:


# Standardize features
features_standardized = standardizer.fit_transform(features)


# In[39]:


# Train a radius neighbors classifier
rnn = RadiusNeighborsClassifier(
radius=.5, n_jobs=-1).fit(features_standardized, target)


# In[40]:


# Create two observations
new_observations = [[ 1, 1, 1, 1]]


# In[41]:


# Predict the class of two observations
rnn.predict(new_observations)


# In[ ]:


features


# ## Explanation

# In[ ]:








