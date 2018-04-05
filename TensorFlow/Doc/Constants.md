### Constants
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
1. **tf.zeros()**     
```tf.zeros(shape, dtype=tf.float32, name=None)```       
creates a tensor of shape and all elements will be zeros (when ran in session)        
2. **tf.zeros_like()**     
```tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)```     
creates a tensor of shape and type (unless type is specified) as the input_tensor but all elements are zeros.       
3. **tf.ones()**     
```tf.ones(shape, dtype=tf.float32, name=None)```    
4. **tf.ones_like()**     
```tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)```      
5. **tf.fill()**      
```tf.fill(dims, value, name=None)```        
creates a tensor filled with a scalar value.       

[Eg zeros](../Codes/Constant_Ex_Zero.ipynb), [Eg ones](../Codes/Constant_Ex_Ones.ipynb), [Eg fill](../Codes/Constant_Ex_Fill.ipynb)        

#### Constants as sequences
1. **tf.linspace()**       
```tf.linspace(start, stop, num, name=None)```       
2. **tf.range()**      
```tf.range(start, limit=None, delta=1, dtype=None, name='range')```       
Tensor objects are not iterable      
```for _ in tf.range(4): # TypeError```     

[Eg Sequence](../Codes/Constant_Ex_Sequence.ipynb)    

#### Randomly Generated Constants
```tf.set_random_seed(seed)```      
```tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)```     
```tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)```     
```tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)```     
```tf.random_shuffle(value, seed=None, name=None)```     
```tf.random_crop(value, size, seed=None, name=None)```     
```tf.multinomial(logits, num_samples, seed=None, name=None)```     
```tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)```     

### What’s wrong with constants?    
Constants are stored in the graph definition. This makes loading graphs expensive when constants are big.    
Therefore, Only use constants for primitive types.    
Use variables or readers for more data that requires more memory


