# Hands-On Machine Learning with Scikit-Learn, Keras, and Tensorflow
## Aurelien Geron
### Read Dec 2023

<details>
<summary><font size=5>Chapter 1: The Machine Learning Landscape</font></summary>


- The ML landscape is the art / science of programming computers to learn from data. They can learn complex relationships between data points to be predictive of new data
- ML is good for when a solution requires a long list of rules, for when traditional approaches yield no good results, fluctuating environments - a ML system can adapt to new data, or getting insights from large amounts of data
- Types of learning
    - Supervised learning - labels are known so the target can be shown to the algorithm. Typical algorithms are KNN, linear reg, logistic reg, SVMs, decision trees & random forests, and NN
    - Unsupervised learning - labels are not known and the model learns for itself. K-means clustering, anomaly detection (one class SVM), anomaly detection (PCA / t-SNE) are all examples
    - Semi-supervised learning algorithms typically combine supervised and unsupervised approaches. So you might first cluster data, and then provide one label for the cluster to learn from, rather than labelling everything
    - Reinforcement learning - this is different entirely, an agent acts in an environment and is given rewards or penalties to act in certain ways
- Batch vs online learning
    - Batch learning - all training is done offline, and generally the entire dataset is used in batches. To retrain a model, you need to retrain using the entire dataset again, and then replace the old model with the new one. Training the new model can take a long time and require lots of resources
    - Online learning - train the model incrementally by feeding it instances of data individually or in mini-batches. The learning step is cheap and fast. But if bad data is fed to the model, your model could quickly lose performance. Hence close monitoring of the model is required
- Main challenges in ML
    - Insufficient training data - I strongly believe more / better data is more important that model selection / tuning
    - Unrepresentative training data
    - Poor quality / noisy data
    - Irrelevant features
    - Overfitting / underfitting training data
- Typical steps in a ML project include: 
    1. Framing the business problem 
    2. Getting and labelling the data 
    3. Analyse and visualise the data to understand what you’re working with
    4. Select a performance metric 
    5. attempt various feature extractions 
    6. get a train and test set 
    7. prototype different model architectures to find one that performs best
    8. tune the best performing model
    9. mature the algorithm
    10. deploy
    11. monitor and maintain
</details>


<details>
<summary><font size=5>Chapter 4: Training Models</font></summary>

- Linear regression is just simply a weighted sum of input features plus a bias
- To implement gradient descent, you compute the gradient of the loss function with regard to each model parameter (input feature), which is the same as computing the partial derivatives wrt each parameter
- With SGD, the result of the gradient of the loss function will bounce up and down because we’re only using a small batch of instances at each step
- You can using training curves to understand model performance when training on different sized subsets of the training set, to see how much of a performance increase you get from adding new data
- High-bias model will typically underfit the data. High-variance model will typically overfit the data. Increasing model complexity will typically increase variance but reduce bias
- Regularisation
    - When you combine both ridge and lasso regularisation, you get ElasticNet regularisation. ElasticNet includes a mix ratio r which controls the amount of ridge and lasso that’s occurring. When r = 0 then you have pure ridge, and when r = 1 you have pure lasso.
        - ElasticNet is better than Lasso which may behave erratically when your training data is wider than is it tall, or when several features are strongly correlated
    - An alternative approach to regularising iterative models is to use early stopping. You stop the training as soon as the validation set has reached a minimum
- A softmax regression can be used for a multiclass problem. You calculate the probability score for each class using a logistic regression. Then with the output probability scores, you run this through a softmax activation function, which computes the exponential of each score, and then normalises them. The predicted class is the one with the highest value
</details>


<details>
<summary><font size=5>Chapter 5: SVMs</font></summary>

- You can think of SVMs as fitting the widest street possible between two support vectors, to classify the dataset on
- In sklearn, you can use hyperparameter C to regularise the SVM model, which controls the number of margin violations that occur
- If your dataset is linear, you can use a linear kernel. But most datasets are more complex than that. You can use non-linear kernels to increase the dimensionality of your dataset
- Scaling your data is important when using SVMs, because a SVM will be able to fit scaled data more easily
- When building an SVM, you should always start by trying a linear kernel. If that doesn’t work then we can try and Gaussian RBF kernel
- SVMs can also be used for regression. In this instance, you flip the objective. So instead of choosing the widest margin that separates the classes, you now try and choose the margin that fits the most instances on the street, without including margin violations
</details>


<details>
<summary><font size=5>Chapter 6: Decision Trees</font></summary>

- Decision trees don’t require any data preparation such as feature scaling or centering
- Sklearn uses CART to train the algorithm, which only produces binary leaves
- You can regularise a decision tree by restricting it’s freedom in training by setting the max depth, max samples in a node needed for splitting, max samples in a lead node, max number of leaf nodes, or max features used for splitting
- Challenges with decision trees
    - They like orthogonal decision boundaries. If you rotate your dataset, you may find the decision boundary becomes convoluted. Using PCA to reduce dimensionality can help with this
    - They’re sensitive to small variations in the training data
</details>


<details>
<summary><font size=5>Chapter 7: Ensemble Learning & Random Forests</font></summary>

- Ensemble algorithms take a combined prediction from multiple trees
- Even if a single model is a weak learner, a combined ensemble may be more predictive than the best performing model in the ensemble
- Bootstrapping is random sampling with replacement from a dataset. Bagging = bootstrap aggregating, ie performing it many times and training an estimator for each bootstrapped dataset
    - When sampling is performed without replacement it’s called pasting
- One way to get a diverse ensemble is to train very different algorithms on the data to create a diverse set of models. Another approach is to train the same model on different random subsets of the data
- Predictors can be trained in parallel on different CPU cores, or on different servers
- Bagging
    - In Sklearn, the BaggingClassifier runs multiple decision tree classifiers, using sampling with replacement
    - Out-of-bag evaluation - this is where the classifier can be evaluated on the training instances that weren’t sampled, because we were using sampling with replacement, this is typically about a third of the dataset. So no separate validation set is needed. Thus you can average the OOB scores for each predictor to get the overall model performance
    - You can also choose to train on a random subset of features each time, useful when training on high dimensionality data
- Random Forests
    - A random forest classifier is the same as a bagging classifier that’s passed a decision tree classifier! But it introduces some extra randomness
        - Rather than always searching for the best parameter on which to split a node, it searches for the best amongst a subset of features. This results in greater tree diversity, which trades a higher bias for a lower variance (ie it reduces overfitting)
    - A great quality of RFs is that they make it easy to understand feature importance, by looking at nodes that use a feature, and seeing by how much it reduces impurity on average
    - RFs can be regularised by tuning the number of estimators
- Boosting
    - The general idea is to make a strong learner from many weak learners, by training predictors sequentially, each trying to correct its predecessor
    - The drawback is that predictors cannot be trained in parallel, so training takes much longer
    - Adaboost
        - Corrects its predecessors by focusing on instances that were underfit, meaning new predictors focus on harder cases
        - It trains a decision tree, then makes predictions on the training set, and increases the relative weight (boosting) of the misclassified instances. The second classifier then trains on the updated weights
        - The learning rate is used to control how much the misclassified instances are boosted. This makes is similar to iteratively performing gradient descent
    - Gradient Boosting (GBRT = gradient boosted regression trees)
        - Works similarly to Adaboost, but instead of training on the updated weighted instances, the successor tries to fit the new model to the residual errors from the previous model
        - The learning rate here is used to scale the contribution of each tree. With a low value, you’ll need more trees to fit the data, but the predictions will likely generalise better. This regularisation technique is called shrinkage
        - You can use early stopping to find the optimal number of trees
    - XGBoost (Extreme Gradient Boosting)
        - XGBoost automatically takes care of early stopping for you
    - Stacking
        - This is a method of using a subsequent model to learn the best combination of predictions from an ensemble, rather than just taking a hard rule like the mean. This could also be called a meta-learner
        - You’d make predictions from the ensemble on a hold-out set of the training data, and then take those predictions and known labels to train a new model (like a linear regressor, or another random forest)
</details>


<details>
<summary><font size=5>Chapter 8: Dimensionality Reduction</font></summary>

- The curse of dimensionality
    - Typically when you have more features than instances, so you have a very high dimensional dataset
    - As you increase dimensions, the distance between 2 points becomes much greater, so much so that each instance looks very different to another instance and it becomes difficult to learn patterns between them
    - You may want to increase the size of your training set, but the number of training instances tends to grow exponentially with the number of features
    - Models will quickly overfit the data if there are too many features
- Projection. Take a swiss roll - rather than squashing the data onto one plane, you want to unroll the swill roll onto a new 2D plane to maintain it’s composition
- Manifold Learning - relies on the hypothesis that most real-world high dimensional datasets lie close to a lower-dimensional manifold. So you need to find what that manifold is.
- Reducing the dimensions of your dataset will generally increase the speed of learning, but may not always result in better performance. It all depends on the dataset
- Principal Component Analysis (PCA)
    - First identifies the hyperplane (flat surface) that lies closest to the data and then projects the data onto it
    - You will typically want to choose a plane that maximises the datasets variance (think of projecting a 2D dataset onto a line and maintaining the width of the distribution). Or you’ll want to minimise the mse between the original dataset and the projection
    - PCA first identifies the axis that gives the largest amount of variance on the training set. And then chooses the axis that’s orthogonal to this to maintain the next largest amount of variance. The number of axes will be equivalent to the number of dimensions
    - You can then reduce down the size of the dataset to d dimensions, but choosing the first d principle components, and projecting the data onto that hyperplane
    - You typically choose the number of dimensions such that you add up to a sufficiently large proportion of the datasets variance (ie 95%)
    - If you perturb the training set a little, then you might get principle components pointing in different directions, but you’d still get a plane that resembles something similar
    - You can also do an inverse transform of the PCA dataset to decompress the data, although it would be identical anymore
    - Incremental PCA allows you to do mini-batch training for PCA, rather than having to fit the entire dataset into memory
    - Kernel PCA performs complex non-linear projections in the original space (rather than linear projects in a high dimensional space). It’s useful for high complex non-linear datasets. kPCA is unsupervised, but you can include it end-to-end in a classification pipeline, and then perform a grid search on the hyperparameters
- t-SNE (t-distributed Stochastic Neighbour Embedding)
    - Keeps similar instances close, and dissimilar instances apart, whilst reducing dimensionality. Mainly used for visualisation
</details>


<details>
<summary><font size=5>Chapter 9: Unsupervised Learning</font></summary>

- Lots of potential here to learn from unlabelled data
- K-Means Clustering
    - Could be used for consumer segmentations, anomaly/fraud detections, or to search for similar images to a reference
    - In k-means, you want to cluster the data into K categories, which they will be labelled with classes. Or alternatively you could return the distance of each instance to the centroid of the clusters, to give a probability of each class
    - You first randomly initialise k centroids in the data, and then label them. Then update the clusters and relabel them iteratively until you converge on a solution
    - Because of the random initialisation thought, you might not always converge on the global minimum. So you typically run the algorithm multiple times, measuring the “inertia” which is the mse to each instances closest centroid.
    - In practice, k-means is more intelligent under the hood, and selects centroids that are far away from each other to reduce the number of times the algorithm needs to run to find an optimal solution
    - Defining the number of clusters is difficult though. Typically you use the mean silhouette coefficient of all the instances. The silhouette coefficient is equal to:
        - $(b-a)/max(a, b)$, where a is the mean distance to other instances in the same cluster, and b is the mean nearest cluster distance
        - It varies between -1 and 1. An instance silhouette coefficient of 1 means it’s well within the cluster, 0 means it’s on the border, and -1 means it’s probably misclassified
        - A silhouette diagram plots the sorted silhouette coefficients for each instance by cluster. You get a knife shape for each cluster. You can use this diagram to check the size of clusters, and may want to choose K even when the mean score is lower
    - Scaling the data is very important before K-Means, otherwise clusters can be very stretched and unpredictable
    - K-means is fast and scalable, but not always strong, particularly for non-spheroidal clusters. Gaussian Mixture Models are better in these cases
    - Clustering can be used for dimensionality reduction and image segmentation by doing colour segmentation
    - A K-means clustering algorithm is typically a hard clustering algorithm, ie it only gives one class to each instance
</details>
