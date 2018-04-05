### Lazy loading
#### What’s lazy loading?

Defer creating/initializing an object until it is needed

#### Normal loading
```
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y) # you create the node for add node before executing the graph

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
    for _ in range(10):
        sess.run(z)
    writer.close()
```

#### Lazy loading:
```
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
    for _ in range(10):
        sess.run(tf.add(x, y)) # someone decides to be clever to save one line of code
    writer.close()
```

Both give the same value of What's the problem?

Imagine you want to compute an op thousands of times! Your graph gets bloated
Slow to load Expensive to pass around.

Solution:
1. Separate definition of ops from computing/running ops
2. Use Python property to ensure function is also loaded once the first time it is called
