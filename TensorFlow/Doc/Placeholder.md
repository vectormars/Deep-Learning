### Feeding Data to the Training Algorithm: Placeholder
A TF program often has 2 phases:
1. Assemble a graph
2. Use a session to execute operations in the graph.

⇒ Can assemble the graph first without knowing the values needed for computation

Analogy:      
Can define the function ```f(x, y) = x*2 + y``` without knowing value of x or y.    
x, y are placeholders for the actual values.

#### Why placeholders?
We, or our clients, can later supply their own data when they need to execute the computation.

#### Placeholders
```tf.placeholder(dtype, shape=None, name=None)```     
* Feed the values to placeholders using a dictionary
* What if want to feed multiple data points in?
    ```
    with tf.Session() as sess:
        for a_value in list_of_values_for_a:
            print sess.run(c, {a: a_value})
    ```

Note:   
If you specify **None** for a dimension, it means "any size".      
[Eg1](../Codes/PlaceHolder_Ex_1.ipynb), [Eg2](../Codes/PlaceHolder_Ex_2.ipynb)     
[Feed data to mini-batch](../Codes/Feed%20data%20to%20mini-batch.ipynb) 

### You can feed_dict any feedable tensor. Placeholder is just a way to indicate that something must be fed.

Use ```tf.Graph.is_feedable(tensor)``` for test. 
Return ```True``` if and only if tensor is feedable.

Eg: Feeding values to TF ops
```
# create operations, tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.mul(a, 3)
with tf.Session() as sess:
# define a dictionary that says to replace the value of 'a' with 15
replace_dict = {a: 15}
# Run the session, passing in 'replace_dict' as the value to 'feed_dict'
sess.run(b, feed_dict=replace_dict) # returns 45
```

**Extremely helpful for testing too**.

