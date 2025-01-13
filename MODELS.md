# Basic Model Operations on PyTorch

# Table of Contents

[TODO]

## What is a model?

In real world, _generally_ if there is a problem, there is also a solution exists. To achieve this solution, mathematical models are created.

**Problem:** What is the slope of the line inside a cartesian coordinate system and how to represent the line formulaically?

**Solution:** Difference in y coordinates divided by difference in x coordinates gives the slope of a line. The line can have a slope that can be represented as `tanθ`. Theta (θ) is the angle between the x axis and the line. A line can also be dislaced for an amount, enabling it to move inside coordinate system.

- This line fitting problem is also called **Linear Regression**

![Slope](resources/slope-calculator-1630062041.png)

- This is what modelling mathematically is. Creating a solution for the problem in a problem space.
- But not all problems can be modelled easily like the slope problem.
- That is when we start to use approximation instead of true answer.
- Approximation has a margin of error unlike solving the modelled equation.
- My professor from college told us once: **"Model it mathematically if you can. That means there is an easier solution for the problem. But if problem is too complex to model mathematically, approximate to the real solution."**
- That is when we use predictive models. That is what underlies today's Machine Learning and Deep Learning methodologies. We approximate and predict.

## How to Create a Model?

- If problem is specific, like finding the slope of points in coordinate system, there should also be a pattern that can be observed. Another example, classifying cats and dogs. As much as both animals have similar properties like having 4 legs, fur and eyes; there are also discriminative properties like fur pattern, whiskers and ear shape. A little bit more complex to model than a simple line but still an existing problem that can be solved.
- If your problem has not have a discriminative pattern, try to reduce your problem to smaller problems that you can start picking up patterns by yourself.

**Note:** Or don't if you want to see correlation between values, categories etc. for academic purposes or such like Exploratory Data Analysis. But there is a strong chance that if there is no pattern to be learned, your model's performance will be as good as guessing randomly.

### Model Creation Preparation 1: Acquire Data

- ML/DL models start with random values and according to some data, those random values adjusted with the values better representing given data.
- So according to that logic first things first we need some data.
- Normally this data would be cleaned by someone and has a pattern that can be learned by ML/DL model.

**Problem:** Find the line that represents given data.

**Solution:** Implement slope function to find suiting.

**Let's create some synthetic data that we can control inputs and outputs by some rule and watch our model learn how to represent it.**

**Expected Output:** Model parameters having close values to ground truth (our synthetic data).

`Linear Regression => f(x) = y = mx + c`

- According to above formula we will generate tensors. But what is what?

**x** is the value in x axis. In this case **X** will be our inputs and will be written as capital by convention as it represents a matrix (a column vector in this case).

**m** is the slope with the x axis. Gives ability to move between 0-90 degrees in the first quadrant of the cartesian coordinate system. This way we can find the line having minimum distance to all points in the data. It is also known as **weight**.

**c** is a constant value. Enables movement in y axis. This value is also known as **bias**.

**y** will be our output/label/prediction feature.

<img src="resources/advancement-made-acquire-data.png" alt="Acquire Data Advancement" style="float:right; margin-bottom: 20px; width: 50%"/>

<div style="clear: both"></div>

```python
import torch

# Can be any arbitrary number.
weight = 0.6
bias = 0.4

# Generate data - X -> features, y -> labels
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1) # add one dimension to turn into column vector.
y = weight * X + bias

X[:5], X.shape, y[:5], y.shape

# First 5 elements of input:
# (tensor([[0.0000],
#          [0.0200],
#          [0.0400],
#          [0.0600],
#          [0.0800]]),

# Size of the input:
# torch.Size([50, 1]),

# First 5 elements of the labels:
# tensor([[0.4000],
#         [0.4120],
#         [0.4240],
#         [0.4360],
#         [0.4480]]),

# Size of the labels:
# torch.Size([50, 1]))
```

### Model Creation Preparation 2: Train Test Split

- I've used 80% of total data as train sample and 20% as test data.

```python
train_size = 0.8 # We are aiming for 80% train and 20% test data.
train_test_split = int(len(X) * train_size)

X_train, y_train = X[:train_test_split], y[:train_test_split]
X_test, y_test = X[train_test_split:], y[train_test_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

# outputs:
# (40, 40, 10, 10)
```

- As we can see data created is a nice line. Inputs between 0 and 1. Due to bias, all labels started from 0.4.

![Synthetic data distribution](resources/synthetic-data-distribution.png)

## How to Create a Model, in PyTorch?

There are 3 main steps to create a model in PyTorch:

1. Subclass `torch.nn.Module`.
2. Set parameters individually or use available layers in constructor.
3. Override `forward()` method fo define what will happen in the forward pass.

### What is Forward Pass?

![forward-pass](resources/forward-pass.png)

For the feedforward neural networks, data flows from input to output. In this process, data is transformed and an output is generated with the help of weights and biasses. Also known as forward propagation.

All `torch.nn.Module` sublasses have to implement the forward function in order to apply transformations to input data.

### Model Structure of Proposed Solution

- All model parameters initialized with random values. This is the start of fitting model to our data.
- Model weight and bias will gradually approach to our synthetic ground truth weight and bias values in order to better represent the data.
- With `nn.Parameter()` class a model parameter can be defined. Has two important parameters within to fill.
  1. data: Initial tensor.
  2. requires_grad: Does tensor require gradient. Used to track gradients for backpropagation.

```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Data has one weight and one bias so one random tensor is required.
        self.weight = nn.Parameter(data=torch.randn(1, dtype=torch.float),
                                    requires_grad=True)

        self.bias = nn.Parameter(data=torch.randn(1, dtype=torch.float),
                                    requires_grad=True)

    # Function for the forward pass.
    # "x" marks the input.
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias
```

### What is Gradient (Descent)?

![gradient](resources/pringles-gradient.png)

_You have seen so many Medium articles at this point I will not use that gradient image. Enjoy the gradient of a pringle._

Gradient descent is the key of machine learning algoritms. Thinking about a hill is overused but the best way to understand it. At the hill top potential is at the maxiumum. Hence can be named as **global maximum**. At the ground level potential is at it's minimum. So it can be called as **global minimum**.

A machine learning algorithm's purpose is to reach the global minimum. Potential is none so no entropy. Basically ideal and prediction result is the same as solving the mathematical model of the problem.

Gradient descent algorithm basically achieves this purpose. There is a gradient (a wiggly surface) and model descents through to reach the ground level.

Key steps are:

1. Start from any point on in gradient.
   - That is why initial parameters are random. Because knowing the whereabouts of global miniumum is near impossible. Starting from anywhere has the same probability of reaching to solution. _I think._
2. Calculate the slope.
   - Using a **loss function** calculate how far the model's predictions from ground truth.
3. Move opposite of the slope
   - Go down hill.
4. Repeat until error/loss minimized.
