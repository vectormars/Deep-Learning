### Data Flow Graphs
In TensorFlow, computation is described using data flow graphs.
* Each **node** of the graph represents an instance of a mathematical operation (like addition, division, or multiplication)
* Each **edge** is a multi-dimensional data set (tensor) on which the operations are performed.
  * **Normal edges**, transfer data structure (tensors) where it is possible that the output of one operation becomes the input for another operation
  * **Special edges**, which are used to control dependency between two nodes to set the order of operation where one node waits for another to finish.

<img src="images/DataFlowGraphs.png" height="350">

A TensorFlow program is typically split into two parts:
1. Construction phase: Build a computation graph 
2. Execution phase: Run the graph and evaluate


##### Constants
```constant(value, dtype=None, shape=None, name='Const', verify_shape=False)```
* ```value``` is an actual constant value which will be used in further computation,
* ```dtype``` is the data type parameter (e.g., float32/64, int8/16, etc.)
* ```shape``` is optional dimensions
* ```name``` is an optional name for the tensor
* ```verify_shape``` is a boolean which indicates verification of the shape of values.      
Ex1: ```z = tf.constant(5.2, name="x", dtype=tf.float32)```      
Ex2: ```a = tf.constant([2, 2], name="a")```      
Ex3: ```b = tf.constant([[0, 1], [2, 3]], name="b")```      
[Eg1](Codes/Constant_Ex_1.ipynb), [Eg2](Codes/Constant_Ex_2.ipynb)    

##### Variable
```k = tf.Variable(tf.zeros([1]), name="k")```    
```y = tf.Variable(x + 5, name='y')```     
Another way to use variables in TensorFlow is in calculations where that variable isn’t trainable and can be defined in the following way
```k = tf.Variable(tf.add(a, b), trainable=False)```   
[Eg1](Codes/Variable_Ex_1.ipynb), [Eg2](Codes/Variable_Ex_2.ipynb), [Eg3](Codes/Variable_Ex_3.ipynb)

##### Sessions
A session encapsulates the control and state of the TensorFlow runtime. A session without parameters will use the default graph created in the current session, otherwise the session class accepts a graph parameter, which is used in that session to be executed.

In order to actually evaluate the nodes, we must run a computational graph within a session.

##### Feeding Data to the Training Algorithm: Placeholder
if you specify **None** for a dimension, it means "any size".      
[Eg1](Codes/PlaceHolder_Ex_1.ipynb), [Eg2](Codes/PlaceHolder_Ex_2.ipynb)

Feed data to mini-batch
```
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size)
    [...] # load data
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch })
            best_theta = theta.eval()
```



##### TensorBoard 
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
Then, ```tensorboard --logdir logs/```. Now TensorBoard is started and running on the default port 6006.             
[Eg1](Codes/TensorBoard_Ex_1.ipynb),[Eg2](Codes/TensorBoard_Ex_2.ipynb)


#### Name scopes
When dealing with more complex models such as neural network, the graph can easily become cluttered with thousands of nodes. To avoid this, you can create *name scopes* tp group related nodes.

#### Mathematics with TensorFlow [(link)](Doc/Mathematics%20with%20TensorFlow.md)



#### Machine Learning with TensorFlow
[Linear Regression](Codes/LinearRegression.ipynb)     
[Gradient Descent](Codes/Batch%20Gradient%20Descent.ipynb)       
[Up_and_running_with_tensorflow](Codes/Up_and_running_with_tensorflow.ipynb)

#### Saving and Restoring Models
* **Saving a model**: Create a **Saver** node
```
[...]
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == :
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch })
    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
```
* **Restoring a model**: 
```
with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
```

Restore only the **theta** variable under the name **weights**:  
```
saver = tf.train.Saver({"weights":theta})
```




Reference:    
https://www.toptal.com/machine-learning/tensorflow-machine-learning-tutorial     
https://www.datacamp.com/community/tutorials/tensorflow-tutorial       
[TensorFlow Math](https://www.tensorflow.org/versions/master/api_guides/python/math_ops#Matrix_Math_Functions)  
