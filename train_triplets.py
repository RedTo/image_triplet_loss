import tensorflow as tf
from preprocessing import PreProcessing
from model import TripletLoss


BATCH_SIZE = 512
TRAIN_ITER = 2000
SAVE_STEP = 50
LEARNING_RATE = 0.01
MOMENTUM = 0.99
MODEL = 'conv_net'
DATA_SRC = '../data/small_rooms_floor_squares_128/'

if __name__ == "__main__":

    # Setup Dataset
    dataset = PreProcessing(DATA_SRC)
    model = TripletLoss()
    placeholder_shape = [None] + list(dataset.images_train.shape[1:])
    print("placeholder_shape", placeholder_shape)

    # Setup Network
    next_batch = dataset.get_triplets_batch
    anchor_input = tf.placeholder(tf.float32, placeholder_shape, name='anchor_input')
    positive_input = tf.placeholder(tf.float32, placeholder_shape, name='positive_input')
    negative_input = tf.placeholder(tf.float32, placeholder_shape, name='negative_input')

    margin = 0.5
    anchor_output = model.conv_net(anchor_input, reuse=False)
    positive_output = model.conv_net(positive_input, reuse=True)
    negative_output = model.conv_net(negative_input, reuse=True)
    loss = model.triplet_loss(anchor_output, positive_output, negative_output, margin)

    # Setup Optimizer
    global_step = tf.Variable(0, trainable=False)

    train_step = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM, use_nesterov=True).minimize(loss, global_step=global_step)

    # Start Training
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Setup Tensorboard
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train.log', sess.graph)

        # Train iter
        for i in range(TRAIN_ITER):
            batch_anchor, batch_positive, batch_negative = next_batch(BATCH_SIZE)

            _, l, summary_str = sess.run([train_step, loss, merged],
                                         feed_dict={
                                         anchor_input: batch_anchor,
                                         positive_input: batch_positive,
                                         negative_input: batch_negative
                                         })

            writer.add_summary(summary_str, i)
            print("\r#%d - Loss" % i, l)

            if (i + 1) % SAVE_STEP == 0:
                saver.save(sess, "model_triplet/model.ckpt")
        saver.save(sess, "model_triplet/model.ckpt")
    print('Training completed successfully.')