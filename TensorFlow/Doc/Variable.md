### Variable
* create variable a with scalar value     
```a = tf.Variable(2, name="scalar")```    
```k = tf.Variable(tf.zeros([1]), name="k")```       
```y = tf.Variable(x + 5, name='y')```        
* create variable b as a vector    
```b = tf.Variable([2, 3], name="vector")```
* create variable c as a 2x2 matrix    
```c = tf.Variable([[0, 1], [2, 3]], name="matrix")```
* create variable W as 784 x 10 tensor, filled with zeros     
```W = tf.Variable(tf.zeros([784,10]))```

Another way to use variables in TensorFlow is in calculations where that variable isn’t trainable and can be defined in the following way      
```k = tf.Variable(tf.add(a, b), trainable=False)```      
[Eg1](../Codes/Variable_Ex_1.ipynb), [Eg2](../Codes/Variable_Ex_2.ipynb), [Eg3](../Codes/Variable_Ex_3.ipynb)

Note: 
```tf.Variable``` is a **class**, but ```tf.constant``` is an **op**    
```tf.Variable``` holds several ops:
```
x = tf.Variable(...)

x.initializer # init op
x.value() # read op
x.assign(...) # write op
x.assign_add(...) # and more
```

#### Initialize your variables
* The easiest way is initializing all variables at once:
    ```
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
    sess.run(init)
    ```
* Initialize only a subset of variables:
    ```
    init_ab = tf.variables_initializer([a, b], name="init_ab")
    with tf.Session() as sess:
        sess.run(init_ab)
    ```
* Initialize a single variable
    ```
    W = tf.Variable(tf.zeros([784,10]))
    with tf.Session() as sess:
        sess.run(W.initializer)
    ```
* Use a variable to initialize another variable
    ```
    W = tf.Variable(tf.truncated_normal([700, 10]))
    U = tf.Variable(2 * W.intialized_value())
    ```
    
#### Evaluate a variable
```
with tf.Session() as sess:
    sess.run(W.initializer)
    print W.eval()
```
[Eg](../Codes/Eval_variable.ipynb)

#### Variable assign
W.assign(100) doesn’t assign the value 100 to W. It creates an assign op, and that op needs to be run to take effect.     
[Assign variable](../Codes/Assign_variable.ipynb)    
[Assign add/sub](../Codes/Assign_add_sub.ipynb)

#### When is a variable initialized? When is it destroyed?
A variable is initialized when you call its initializer, and it is destroyed when the session ends. In distributed TensorFlow, variables live in containers on the cluster, so closing a session will **not** destroy the variable. To destroy a variable, you need to clear its container.

