### Manage your model
#### Part I. Save model
Use ```tf.train.Saver``` saves graph’s variables in binary files       
Note: Saves variables, not graphs!       
```tf.train.Saver.save(sess, save_path, global_step=None...)```      

Eg: Save parameters after 1000 steps
```
if (step + 1) % 1000==0:
    saver.save(sess, 'checkpoint_directory/model_name', global_step=model.global_step)
```

**Global step**:      
```self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')```      
```self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)```

#### Part 2. Restore 
* **Restore variables**       
```saver.restore(sess, 'checkpoints/name_of_the_checkpoint')```  
* **Restore latest checkpoint**
```ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))```       
```
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
```
    1. checkpoint keeps track of the latest checkpoint
    2. Safeguard to restore checkpoints only when there are checkpoints

[Eg](../Codes/Saving%20and%20restoring%20a%20model.ipynb)
