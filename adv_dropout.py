import tensorflow as tf
import numpy as np

tensor = tf.constant(np.repeat([1, 2, 3, 4, 5, 6, 7], 4))

#initialize the variable
init_op = tf.initialize_all_variables()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(tensor))