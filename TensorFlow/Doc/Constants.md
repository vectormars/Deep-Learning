#### Constants
```constant(value, dtype=None, shape=None, name='Const', verify_shape=False)```
* ```value``` is an actual constant value which will be used in further computation,
* ```dtype``` is the data type parameter (e.g., float32/64, int8/16, etc.)
* ```shape``` is optional dimensions
* ```name``` is an optional name for the tensor
* ```verify_shape``` is a boolean which indicates verification of the shape of values.      
Ex1: ```z = tf.constant(5.2, name="x", dtype=tf.float32)```      
Ex2: ```a = tf.constant([2, 2], name="a")```      
Ex3: ```b = tf.constant([[0, 1], [2, 3]], name="b")```      
[Eg1](../Codes/Constant_Ex_1.ipynb), [Eg2](../Codes/Constant_Ex_2.ipynb), [Eg3](../Codes/Constant_Ex_3.ipynb)   

#### Tensors filled with a specific value
1. **Zeros**     
```tf.zeros(shape, dtype=tf.float32, name=None)```
