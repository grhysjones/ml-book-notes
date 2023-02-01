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
