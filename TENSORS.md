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
