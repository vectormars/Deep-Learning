#### Mathematics with TensorFlow
**Tensors** are the basic data structures in TensorFlow, and they represent the connecting **edges** in a dataflow graph.

A tensor simply identifies a multidimensional array or list. The tensor structure can be identified with three parameters: rank, shape, and type.

* **Rank**: Identifies the number of dimensions of the tensor. A rank is known as the order or n-dimensions of a tensor, where for example rank 1 tensor is a vector or rank 2 tensor is matrix.
* **Shape**: The shape of a tensor is the number of rows and columns it has.
* **Type**: The data type assigned to tensor elements.

To build a tensor in TensorFlow, we can build an n-dimensional array. This can be done easily by using the NumPy library, or by converting a Python n-dimensional array into a TensorFlow tensor.

<img src="images/tensors.png" height="350">

[Eg1](Codes/Build%20a%201-d%20tensor.ipynb): Build a 1-d tensor, by using a NumPy array   
[Eg2](Codes/Build%20a%202-d%20tensor.ipynb): Build a 2-d tensor(matrix), by using a NumPy array   

##### Tensor Operations

| TensorFlow operator | Description |
|---------------------|-------------|
| tf.add              | x+y         |
| tf.subtract         | x-y         |
| tf.multiply         | x*y         |
| tf.div              | x/y         |
| tf.mod              | x % y       |
| tf.abs              | abs(x)       |
| tf.negative         | -x          |
| tf.sign             | sign(x)     |
| tf.square           | x*x         |
| tf.round            | round(x)    |
| tf.sqrt             | sqrt(x)     |
| tf.pow              | x^y         |
| tf.exp              | e^x         |
| tf.log              | log(x)      |
| tf.maximum          | max(x, y)   |
| tf.minimum          | min(x, y)   |
| tf.cos              | cos(x)      |
| tf.sin              | sin(x)      |

TensorFlow operations listed in the table above work with tensor objects, and are performed **element-wise**. So if you want to calculate the cosine for a vector x, the TensorFlow operation will do calculations for each element in the passed tensor.      
```    
tensor_1d = np.array([0, 0, 0])
tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)
with tf.Session() as session:
    print(session.run(tf.cos(tensor)))
```     
Output: ```[ 1.  1.  1.]```

##### Matrix Operations
TensorFlow supports all the most common matrix operations, like multiplication, transposing, inversion, calculating the determinant, solving linear equations, and many more.   
[Eg](Codes/Matrix%20Operation%201.ipynb)
   
##### Reduction
TensorFlow supports different kinds of reduction. Reduction is an operation that removes one or more dimensions from a tensor by performing certain operations across those dimensions. A list of supported reductions for the current version of TensorFlow can be found here. We will present a few of them in the [example](Codes/Reduction_Ex_1.ipynb).

##### Segmentation
Segmentation is a process in which one of the dimensions is the process of mapping dimensions onto provided segment indexes, and the resulting elements are determined by an index row.

Segmentation is actually grouping the elements under repeated indexes.

<img src="images/Segmentation.png" height="350">

[Eg](Codes/Segmentation_Ex_1.ipynb)

##### Sequence Utilities
Sequence utilities include methods such as:
* **argmin** function, which returns the index with min value across the axes of the input tensor,
* **argmax** function, which returns the index with max value across the axes of the input tensor,
* **setdiff**, which computes the difference between two lists of numbers or strings,
* **where** function, which will return elements either from two passed elements x or y, which depends on the passed condition
* **unique** function, which will return unique elements in a 1-D tensor.

[Eg](Codes/Sequence_Utilities_Ex_1.ipynb)
