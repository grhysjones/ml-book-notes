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
