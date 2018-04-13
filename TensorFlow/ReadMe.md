### Data Flow Graphs
In TensorFlow, computation is described using data flow graphs.
* Each **node** of the graph represents an instance of a **mathematical operation** (like addition, division, or multiplication)
* Each **edge** is a multi-dimensional **data set** (tensor) on which the operations are performed.
  * **Normal edges**, transfer data structure (tensors) where it is possible that the output of one operation becomes the input for another operation
  * **Special edges**, which are used to control dependency between two nodes to set the order of operation where one node waits for another to finish.

<img src="images/DataFlowGraphs.png" height="350">

Main benefits and drawbacks of creating a computation graph rather than directly executing the computation:
* Main benefits     
  * TensorFlow can automatically compute the gradients for you (use the reverse-mode autodiff)
  * TensorFlow can take care of running the operations in parallel in different threads
  * It makes it easier to run the same model across different devices
  * It simplifier introspection-for example, to view the model in TensorBoard
* Main drawbacks:
  * It makes the learning curve steeper
  * It makes step-by-step debugging harder


### What’s a tensor?
An n-dimensional array    
* 0-d tensor: scalar (number)
* 1-d tensor: vector
* 2-d tensor: matrix
* and so on
<img src="images/tensors.png" height="350">

### TensorFlow program
A TensorFlow program is typically split into two parts:
1. **Construction phase**: Assemble a graph      
2. **Execution phase**: Use a **session** to execute operations in the graph.

#### Sessions
* A session encapsulates the control and state of the TensorFlow runtime. A session without parameters will use the default graph created in the current session, otherwise the session class accepts a graph parameter, which is used in that session to be executed.
* In order to actually evaluate the nodes, we must run a computational graph within a session.
* Each session maintains its own copy of variable
* Session vs InteractiveSession      
  You sometimes see InteractiveSession instead of Session. The only difference is an InteractiveSession makes itself the default       
    ```
    sess = tf.InteractiveSession()
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    c = a * b
    # We can just use 'c.eval()' without specifying the context 'sess'
    print(c.eval())
    sess.close()
    ```

#### TensorFlow Data Types
* TensorFlow takes Python natives types: boolean, numeric (int, float), strings   
* TensorFlow integrates seamlessly with **NumPy** 
* **Do not** use Python native types for tensors because TensorFlow has to infer Python type

#### Constants [(link)](Doc/Constants.md)
```tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)```
#### Variable [(link)](Doc/Variable.md)
```
tf.Variable(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None)
```

tf.constant and tf.Variable:
* Constant values are stored in the graph definition
* Sessions allocate memory to store variable values

#### Placeholder and feed_dict [(link)](Doc/Placeholder.md)
```tf.placeholder(dtype, shape=None, name=None)```
* Feed values into placeholders by dictionary (feed_dict)
* You can feed values in variables too


#### Lazy loading [(link)](Doc/Lazy%20loading.md)  

#### TensorBoard 
TensorBoard is a visualization tool for analyzing data flow graphs. This can be useful for gaining better understanding of machine learning models.
```
import tensorflow as tf

x = tf.constant(-2.0, name="x", dtype=tf.float32)
a = tf.constant(5.0, name="a", dtype=tf.float32)
b = tf.constant(13.0, name="b", dtype=tf.float32)

y = tf.Variable(tf.add(tf.multiply(a, x), b))

init = tf.global_variables_initializer()

with tf.Session() as session:
    merged = tf.summary.merge_all() // new
    writer = tf.summary.FileWriter("logs", session.graph) // new

    session.run(init)
    print(session.run(y))

writer.close()
```
Then, ```tensorboard --logdir logs/ --host=127.0.0.1```. Now TensorBoard is started and running on the default port 6006 (Type localhost:6006 in chrome).                  
[Eg1](Codes/TensorBoard_Ex_1.ipynb), [Eg2](Codes/TensorBoard_Ex_2.ipynb), [Eg3](Codes/TensorBoard_Ex_3.ipynb)


#### Name scopes (Group nodes together)
When dealing with more complex models such as neural network, the graph can easily become cluttered with thousands of nodes. To avoid this, you can create *name scopes* tp group related nodes.     
```with tf.name_scope("my_name") as my_name:```      
[Eg](Codes/Name%20scopes.ipynb)

#### Mathematics with TensorFlow [(link)](Doc/Mathematics%20with%20TensorFlow.md)

####  Modularity
Avoid repetitive code    
[Eg](Codes/Modularity.ipynb)

#### Sharing variable
* If you want to share a variable between various components of your graph, one simple option is to create it first, then pass it as a parameter to the functions that need it.
* Another option is to set the shared variable as an attribute of the function upon the first call.    
[Eg](Codes/Sharing%20Variables.ipynb)

#### Saving and Restoring Models
* **Saving a model**: Create a **Saver** node
* **Restoring a model**    
[Eg](Codes/Saving%20and%20restoring%20a%20model.ipynb)


#### List of optimizers in TF
1. tf.train.GradientDescentOptimizer
2. tf.train.AdagradOptimizer
3. tf.train.MomentumOptimizer
4. tf.train.AdamOptimizer
5. tf.train.ProximalGradientDescentOptimizer
6. tf.train.ProximalAdagradOptimizer
7. tf.train.RMSPropOptimizer
8. And more


## Overall structure of a model in TensorFlow
### Phase 1: Assemble graph
1. Define placeholders for input and output
2. Define the weights
3. Define the inference model
4. Define loss function
5. Define optimizer

### Phase 2: Execute the computation
1. Initialize all model variables for the first time.
2. Feed in the training data. Might involve randomizing the order of data samples.
3. Execute the inference model on the training data, so it calculates for each training input
example the output with the current model parameters.
4. Compute the cost
5. Adjust the model parameters to minimize/maximize the cost depending on the model.



## Machine Learning Examples in TensorFlow
[Linear Regression 1](Codes/Linear%20regression%20Stanford%20CS20.ipynb)     
[Linear Regression 2](Codes/LinearRegression.ipynb)     
[Logistic Regression 1](Codes/Logistic%20regression%20Stanford%20CS20.ipynb)      
[Gradient Descent](Codes/Batch%20Gradient%20Descent.ipynb)       
[Up_and_running_with_tensorflow](Codes/Up_and_running_with_tensorflow.ipynb)



## Reference:    
https://www.toptal.com/machine-learning/tensorflow-machine-learning-tutorial     
https://www.datacamp.com/community/tutorials/tensorflow-tutorial       
[TensorFlow Math](https://www.tensorflow.org/versions/master/api_guides/python/math_ops#Matrix_Math_Functions)  
