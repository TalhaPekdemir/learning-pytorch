# Classification with PyTorch

There are 3 main types in Machine Learning:

1. Supervised Learning
   - The ML algorithm is fed with both the data and corresponding labels.
2. Unsupervised Learning
   - No label provided with the data. Useful for finding patterns and relations in data.
3. Reinforcement Learning
   - The model is trained interacting with environment and either rewarded of punished for actions. Used in fields like simulations, strategy and robotics.

**Note:** This repo and this document currently focusing on **Supervised Learning**.

<hr>

Supervised Learning has 2 main subcategories within:

1. [Regression](MODELS.md)
   - Model prediction is a continuous value. Ex. Linear Regression, Polynomial Regression.
2. Classification
   - Model prediction is categorical, a discrete value. Ex. Logistic regression, Support Vector machine.

Classification problems tackled here as a Deep Learning problem rather than a classic Machine Learning problem.

Why? Although, PyTorch is capable of modelling Machine Learning problems, actually a Deep Learning library. Notice that we are solving problems using layers, which are neural networks used in Deep learning. So we are actually solving problems using Deep learning techniques. You might say that, But the calculations underneath layers are actually equations from Machine Learning and you have proven that in [Models](MODELS.md) section with a regression model. You are right. But do not forget that Deep Learning is a subcategory of Machine Learning field.

Moving on from here, which is classification problems, terms and algorithms used will be evaluated in terms of Deep Learning. Still, many things will be similar to what we have done before. Actually almost everything will be same due to PyTorch's abstraction of algorithms.

<hr>

Classification in Deep Learning has 3 main subcategories:

1. [Binary Classification](CLASSIFICATION.BINARY.md)
   - Model output is 0 or 1. True or false. It is or it is not. Ex. fraud detection, spam detection.
2. [Multiclass Classification](CLASSIFICATION.MULTICLASS.md)
   - Model output can be 0, 1, 2 corresponding to cat, dog and alligator. Or any amount of classes ranging from 0 to n. Ex. image classification.
3. Multilabel Classification
   - Output can be multiple classes like tags on a forum of Wikipedia. Ex. movie genre classifier/recommender.
