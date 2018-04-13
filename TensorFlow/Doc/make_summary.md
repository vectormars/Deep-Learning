### Make summary
```tf.summary```      
Visualize our summary statistics during our training
* tf.summary.scalar
* tf.summary.histogram
* tf.summary.image

**Step 1**: create summaries
```
with tf.name_scope("summaries"):
    tf.summary.scalar("loss", self.loss)
    tf.summary.scalar("accuracy", self.accuracy)
    tf.summary.histogram("histogram loss", self.loss)
    
    # merge them all
    self.summary_op = tf.summary.merge_all()
```

**Step 2**: run them
```
loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)
```
Like everything else in TF, summaries are ops

**Step 3**: write summaries to file       
```writer.add_summary(summary, global_step=step)```
