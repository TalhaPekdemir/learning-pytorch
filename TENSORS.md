# Tensors on PyTorch

For more info refer to the [docs](https://pytorch.org/docs/stable/tensors.html).

- A tensor can hold only numerical values like whole numbers (eg. int64, int32) and floating point numbers (eg. float32, float16).

- A tensor can encapsulate a single value, an array of values or values in multiple dimensions.

  **Tensor special names**

  | Value Count | Special Name |
  | ----------- | ------------ |
  | 1           | scalar       |
  | N           | vector       |
  | NxM         | matrix       |
  | NxMxdim     | tensor       |

## Creation of Tensor

### By Hand:

```python
import torch

# 0D value
scalar = torch.tensor(1)
print(scalar) # tensor(1)

# 1D value
vector = torch.tensor([1, 2, 3])
print(vector) # tensor([1, 2, 3])

# 2D value
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

print(matrix) # tensor([[1, 2, 3],
#                       [4, 5, 6],
#                       [7, 8, 9]])

# 3D value
tensor = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]],
                       [[10, 11, 12],
                        [13, 14, 15],
                        [16, 17, 18]]])

print(tensor) # tensor([[[1, 2, 3],
#                        [4, 5, 6],
#                        [7, 8, 9]],
#                       [[10, 11, 12],
#                        [13, 14, 15],
#                        [16, 17, 18]]])
```

### With Methods

Common ways to initialize tensor with methods are:

#### Random Tensor

```python
random_tensor = torch.rand(2)

# tensor([0.3268, 0.5073])
```

```python
# with dimesional size parameter
random_tensor = torch.rand(size=(3, 2))

# tensor([[0.2733, 0.3337],
#         [0.4840, 0.6281],
#         [0.9345, 0.2432]])
```

#### Tensor of Zeros

```python
zeros_tensor = torch.zeros(size=(5, 5))

# tensor([[0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])
```

#### Tensor of Ones

```python
ones_tensor = torch.ones(size=(2, 2))

# tensor([[1., 1.],
#         [1., 1.]])
```

- **Why ones and zeros?**: Useful for masking opearations.

#### Tensor in Range

```python
# [start, end)
range_tensor = torch.arange(start=1, end=10)

# tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
```

```python
even_tensor = torch.arange(start=0, end=10, step=2)

# tensor([0, 2, 4, 6, 8])
```

#### New Tensor from Shape of Another Tensor

```python
zeros_like_tensor = torch.zeros_like(input=range_tensor)

# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
```

```python
ones_like_tensor = torch.ones_like(input=range_tensor)

# tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
```

### Intermediate Tensor Creation

- `torch.tensor` has 3 base parameters to pay attention.

  ```python
  params_tensor = torch.tensor(data=[1, 9, 0, 3],
                              dtype=None,
                              device=None,
                              requires_grad=False)
  ```

#### Tensor Data Type

- Default tensor dtype is `torch.float32` in PyTorch. Can be checked like `torch.get_default_dtype()`.

- But if at the creation time tensor dtype not specified (`dtype=None`), dtype will be inferred depending on the tensor data.

  ```python
  import torch

  int_tensor = torch.tensor(data=[1, 2, 3], dtype=None)

  print(int_tensor.dtype) # torch.int64

  float_tensor = torch.tensor(data=[1.5, 5,7], dtype=None)

  print(float_tensor.dtype) # torch.float32
  ```

#### Tensor Device

- If not specified, tensors live on memory. If specified tensors can be sent to GPU VRAM for faster and parallelized calculation. Defaults to `"cpu"`.

```python
# to create tensor in GPU memory
params_tensor = torch.tensor(data=[1, 9, 0, 3],
                             dtype=None,
                             device="cuda",
                             requires_grad=False)
```

[TODO] requires_grad

## Tensor Properties

| Property      | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| tensor.item() | Get the data inside tensor as Python types.                  |
| tensor.data   | Returns tensor data as tensor type                           |
| tensor.device | Returns the device tensor is living. (eg. cpu or cuda)       |
| tensor.dtype  | Returns tensor data type (eg. torch.float32)                 |
| tensor.ndim   | Get the dimension info of tensor.                            |
| tensor.shape  | Get the shape of tensor. Returns an array of dimesion sizes. |
| tensor.size() | Returns tensor shape. Similar to `tensor.shape`              |

- An easy way of telling the exact dimension of a tensor by looking, simply count square brackets at the beginning.

## Tensor Manipulation

- Basic operations like addition, subtraction, multiplication and division are made element-wise. Meaning left operand is `i`th element of Tensor A and right operand is `i`th element of Tensor B.

- Tensors can be subjected to mathematical operations with constants. This will have an effect of constant being used as operand for each item in tensor.

- Another essential but widely used (probably the most) operation is [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication).

- Instead of each element generating one result, entire `i`th row and entire `i`th column contributes to one result. This operation called as [dot product](https://en.wikipedia.org/wiki/Dot_product).

```python

```

### Addition

```python
# Constants
op_tensor = torch.tensor([1, 2, 3])

# With python operator
op_tensor + 5                 # tensor([6, 7, 8])

# With torch method
torch.add(op_tensor, 5)       # tensor([6, 7, 8])

# Another tensor
tensor_A = torch.tensor([0, 2, 4])
tensor_B = torch.tensor([1, 3, 5])

tensor_A + tensor_B           # tensor([1, 5, 9])

torch.add(tensor_A, tensor_B) # tensor([1, 5, 9])
```

### Subtraction

```python
# Constants
op_tensor = tensor([1, 3, 5])

# With python operator
op_tensor - 10                # tensor([-9, -8, -7])

# With torch method
torch.sub(op_tensor, 10)      # tensor([-9, -8, -7])

# Another tensor
tensor_A = torch.tensor([0, 2, 4])
tensor_B = torch.tensor([1, 3, 5])

tensor_A - tensor_B           # tensor([-1, -1, -1])

torch.sub(tensor_A, tensor_B) # tensor([-1, -1, -1])
```

### Multiplication

```python
# Constants
op_tensor = tensor([1, 3, 5])

# With python operator
op_tensor * 8                 # tensor([ 8, 16, 24])

# With torch method
torch.mul(op_tensor, 10)      # tensor([ 8, 16, 24])

# Another tensor
tensor_A = torch.tensor([0, 2, 4])
tensor_B = torch.tensor([1, 3, 5])

tensor_A * tensor_B           # tensor([0, 6, 20])

torch.mul(tensor_A, tensor_B) # tensor([0, 6, 20])
```

### Division

```python
# Constants
op_tensor = tensor([1, 3, 5])

# With python operator
op_tensor / 2                 # tensor([0.5000, 1.0000, 1.5000])

# With torch method
torch.div(op_tensor, 10)      # tensor([0.5000, 1.0000, 1.5000])

# Another tensor
tensor_A = torch.tensor([0, 2, 4])
tensor_B = torch.tensor([1, 3, 5])

tensor_A / tensor_B           # tensor([0.0000, 0.6667, 0.8000])

torch.div(tensor_A, tensor_B) # tensor([0.0000, 0.6667, 0.8000])
```

### Matrix Multiplication

**Note:** `@` symbol used for dot product.

Rules:

1. Inner dimensions should be the same.

   ```
   (5, 2) @ (3, 12) -> won't work

   (5, 2) @ (2, 12) -> will work
   ```

2. Multiplication result will have the shape of outer dimensions.

   `(5,2) @ (2, 12) -> (5, 12)`

```python
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])

# Would give an error: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2) because iner dimension for tensor_A is 2 but inner shape for tensor_B is 3.
torch.matmul(tensor_A, tensor_B)
```

```python
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])

tensor_B = torch.tensor([[7, 8, 9],
                         [10, 11, 12]])

torch.matmul(tensor_A, tensor_B)
# tensor([[ 27,  30,  33],
#         [ 61,  68,  75],
#         [ 95, 106, 117]])
```

### Matrix Transpose

- It is like turning a matrix in `y=-x` axis. Think about holding from upper left corner of the matrix with your left hand and lower right corner of the matrix with your right hand. Then turn it bottom to top (or top to bottom).

[A cool site for visualizing transpose](https://matrix.reshish.com/transpose.php)

![matrix_transpose_visualization.png](resources/matrix_transpose_visualization.png)

- Useful when tensor is row aligned or vice-versa. Or turning a row vector to column vector and also vice-versa. Why vice-versa? Because transposing a transposed matrix gives the original matrix.

```python
matrix = torch.tensor([[7, 10],
                       [8, 11],
                       [9, 12]])

matrix.T
# tensor([[ 7,  8,  9],
#         [10, 11, 12]])

print("matrix shape:", tensor_B.T.shape)
print("transposed matrix shape:", tensor_B.T.shape)

# Output:
# matrix shape: torch.Size([3, 2])
# transposed matrix shape: torch.Size([2, 3])
```

## Tensor Aggregation

```python
agg_tensor = torch.arange(1, 10)
agg_tensor, agg_tensor.dtype

# Output:
# (tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]), torch.int64)
```

### Min

Returns the min value in tensor.

```python
# Either use tensor function directly or use torch function and pass the tensor
agg_tensor.min(), torch.min(agg_tensor)

# Output:
# (tensor(1), tensor(1))
```

### Max

Returns the maximum value in tensor.

```python
agg_tensor.max(), torch.max(agg_tensor)

# Output:
# (tensor(9), tensor(9))
```

### Mean

Returns the average of values in tensor. 

```python
# Mean excepts a float or complex number input
# https://pytorch.org/docs/stable/generated/torch.mean.html
agg_tensor.mean()

# Either use optional dtype parameter to specify both dtype of returned tensor
# and cast input tensor before operation
print("With tensor method: ", agg_tensor.mean(dtype=torch.float32))

# Or cast it before using if not sure.
print("With tensor method but input casted before:", 
      agg_tensor.type(torch.float32).mean())

# Similar to min and max, mean can be calculated with torch methods.
# Beware that input tensor still needed to be casted to suppoted dtype.
print("With torch methods: ", torch.mean(agg_tensor.type(torch.float32)))

# Output:
# With tensor method:  tensor(5.)
# With tensor method but input casted before: tensor(5.)
# With torch methods:  tensor(5.)
```

### Sum

Returns the sum of values in tensor.

```python
agg_tensor.sum(), torch.sum(agg_tensor)

# Output:
# (tensor(45), tensor(45))
```

### Arg Min

Returns the index o miniumum value in tensor.

```python
agg_tensor.argmin(), torch.argmin(agg_tensor)

# Output:
# (tensor(0), tensor(0))
```

### Arg Max

Returns the index o maximum value in tensor.

```python
agg_tensor.argmax(), torch.argmax(agg_tensor)

# Output: 
# (tensor(8), tensor(8))
```

## Tensor Shape Manipulation

- Reshape: Reshape tensor to specified shape.
- View: Using the same memory return a view of the tensor.
- Stack: Stack tensor on top of each other (vertical stack) or side by side (horizonal stack). vstack and hstack also exists as seperate methods.
- Squeeze: Removes the shape of `1` dimensions from tensor.
- Unsqueeze: Adds a new dimension of `1` to specified tensor index.
- Permute: Rearrange dimension order.

```python
dummy_tensor = torch.arange(1, 13)
dummy_tensor, dummy_tensor.shape

# Outputs:
# (tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]), torch.Size([12]))
```

### Reshape

- New shape must be compatible with old tensor's shape

```python
reshaped_tensor = dummy_tensor.reshape(1, 10)

# Outputs:
# shape '[1, 10]' is invalid for input of size 12
```

```python
reshaped_tensor = dummy_tensor.reshape(1, 12)
reshaped_tensor, reshaped_tensor.shape

# Outputs:
# (tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]]), torch.Size([1, 12]))
```

- A row vector can be turned into a column vecotor with reshaping.

```python
reshaped_tensor = dummy_tensor.reshape(12, 1)
reshaped_tensor, reshaped_tensor.shape

# Outputs:
# (tensor([[ 1],
#          [ 2],
#          [ 3],
#          [ 4],
#          [ 5],
#          [ 6],
#          [ 7],
#          [ 8],
#          [ 9],
#          [10],
#          [11],
#          [12]]),
#  torch.Size([12, 1]))
```

```python
reshaped_tensor = dummy_tensor.reshape(4, 3)
reshaped_tensor, reshaped_tensor.shape

# Outputs:
# (tensor([[ 1,  2,  3],
#          [ 4,  5,  6],
#          [ 7,  8,  9],
#          [10, 11, 12]]),
#  torch.Size([4, 3]))
```


### View

- Returns a reshaped tensor based from another tensor. Viewed tensor shares the same memory with original tensor's values.

- Original tensor is not reshaped but if a value changed in view tensor, same value change will occur in original tensor.

```python
view_tensor = dummy_tensor.view(3,4)
# View tensor:  
# tensor([[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]])
#
# Original tensor:  
# tensor([1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

view_tensor[0, 0] = 99 # make changes

# View tensor:  
# tensor([[99,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]])

# Original tensor:  
# tensor([99,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

```

### Stack

- Only tensors with same size can be stacked with this method.

- Stacking happens according to the index of input tensor sequence.

- dim parameter defaults to 0. `0` for stacking as rows, `1` for stacking as columns.

```python
stacked_tensor = torch.stack([dummy_tensor, dummy_tensor], dim=0)

# stacked_tensor:
# tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#         [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]])

```

```python
stacked_tensor = torch.stack([dummy_tensor, dummy_tensor], dim=1)

# stacked_tensor:
# tensor([[ 1,  1],
#         [ 2,  2],
#         [ 3,  3],
#         [ 4,  4],
#         [ 5,  5],
#         [ 6,  6],
#         [ 7,  7],
#         [ 8,  8],
#         [ 9,  9],
#         [10, 10],
#         [11, 11],
#         [12, 12]])
```

### hstack

- hstack method concatenates the inputs side by side depending on the written order. If row counts of tensors are equal, all's good with the world.
- If two tensors are vectors, next one will be appended to the previous tensor.

```python
hstack_dummy = torch.arange(1, 5)

hstack_mat_A = torch.arange(1, 5).reshape((2,2))
hstack_mat_B = torch.arange(5, 11).reshape((2,3)) # this works
#hstack_mat_B = torch.arange(5, 11).reshape((3,2)) # this won't work

print(hstack_mat_A)
print(hstack_mat_B)

print(torch.hstack([hstack_mat_A, hstack_mat_B]))

# matrix A:
# tensor([[1, 2],
#         [3, 4]])
#
# matrix B:
# tensor([[ 5,  6,  7],
#         [ 8,  9, 10]])
#
# hstacked matrix:
# tensor([[ 1,  2,  5,  6,  7],
#         [ 3,  4,  8,  9, 10]])
```

```python
hstack_tensor_A = torch.arange(1, 9).reshape((2, 2, 2))
hstack_tensor_B = torch.arange(9, 17).reshape((2, 2, 2))

print("Tensor A:", hstack_tensor_A)
print("Tensor B", hstack_tensor_B)

print(torch.hstack([hstack_tensor_A, hstack_tensor_B]))

# Tensor A: 
# tensor([[[1, 2],
#          [3, 4]],

#         [[5, 6],
#          [7, 8]]])
# Tensor B: 
# tensor([[[ 9, 10],
#          [11, 12]],

#         [[13, 14],
#          [15, 16]]])

# Resulting tensor:
# tensor([[[ 1,  2],
#          [ 3,  4],
#          [ 9, 10],
#          [11, 12]],

#         [[ 5,  6],
#          [ 7,  8],
#          [13, 14],
#          [15, 16]]])
```

![hstack_logic.png](resources/hstack_logic.png)

### vstack

- This one does not have much of an appeal. Similar to stack(dim=0).

```python
torch.vstack([dummy_tensor, dummy_tensor])

# Outputs:
# tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#         [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]])
```

### Squeeze

- If you have a tensor of shape (1, 224, 224, 3) but just (224, 224, 3) required, Just squeeze it. Drops the dimensions with rank `1`.
- Also supports removing from specified dimensions using tuple for all specified rank `1` dims or int if single dim to be removed.

```python
sq_tensor = torch.zeros((2, 3, 1)) # depth, col, row

print(sq_tensor)
sq_tensor.squeeze(), torch.squeeze(sq_tensor)

# Original tensor:
# tensor([[[0.],
#          [0.],
#          [0.]],

#         [[0.],
#          [0.],
#          [0.]]])

# Squeezed with tensor method.
# (tensor([[0., 0., 0.],
#          [0., 0., 0.]]),
#
# Squeezed with torch method.
#  tensor([[0., 0., 0.],
#          [0., 0., 0.]]))
```

```
    +---+
   /   /|
  +---+ |                           +---+---+---+
 /   /| +                          /   /   /   /|
+---+ |/|                         +---+---+---+ |
|   | + |              \          |   |   |   | +
|   |/| +         ------\         |   |   |   |/|
+---+ |/|         ------/         +---+---+---+ |
|   | + |              /          |   |   |   | + 
|   |/| +                         |   |   |   |/
+---+ |/                          +---+---+---+
|   | +   
|   |/    
+---+ 
```

### Unsqueeze

- Adds a rank `1` dimension to a given dimension index.

```python
usq_tensor = torch.arange(1, 10)

print(f"""
Shape of tensor before unsqueezing: {usq_tensor.shape}
Shape of tensor after unsqueezing for index 0: {usq_tensor.unsqueeze(dim=0).shape}
Shape of tensor after unsqueezing for index 1: {usq_tensor.unsqueeze(dim=1).shape}
""")

# Outputs:
# Shape of tensor before unsqueezing: torch.Size([9])
# Shape of tensor after unsqueezing for index 0: torch.Size([1, 9])
# Shape of tensor after unsqueezing for index 1: torch.Size([9, 1])
```

### Permute

- To change the order of dimensions of a tensor. Works as view, hence tensor values are in same memory space.

```python
#                           0    1   2
image_tensor = torch.rand((128, 128, 3))
print("Image is currently in order of width, height, color channels with the size of", image_tensor.shape)

rearranged_tensor = torch.permute(image_tensor, dims=(2, 1, 0)) # 2->0, 1->1, 0->2
print("Image rearranged as color chanels, width, height. New shape is ", rearranged_tensor.shape)

# Outputs:
# Image is currently in order of width, height, color channels with the size of torch.Size([128, 128, 3])
# Image rearranged as color chanels, width, height. New shape is torch.Size([3, 128, 128])
```

- Changes on assigned variable will effect original tensor since permute works as view.

```python
print("image_tensor before changing permuted tensor:", image_tensor[0, 0, 0])
rearranged_tensor[0, 0, 0] = 1

print("image_tensor after changing permuted tensor:", image_tensor[0, 0, 0])

# Outputs:
# image_tensor before changing permuted tensor: tensor(0.4575)
# image_tensor after changing permuted tensor: tensor(1.)
```