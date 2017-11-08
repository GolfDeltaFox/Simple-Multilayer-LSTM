import numpy as np
import tensorflow as tf
import random
from dataset import data_generator
from initializer import initializer

initializer()

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_graph(
    state_size = 100,
    nb_class = 10,
    batch_size = 32,
    dropout_keep = 1.0,
    num_steps = 200,
    num_layers = 3,
    build_with_dropout=False,
    learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    # batch x num_steps x depth
    rnn_inputs = tf.one_hot(x, nb_class, axis=-1)

    stacked_rnn = []
    for i in range(num_layers):
        new_cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)
        new_cell = tf.nn.rnn_cell.DropoutWrapper(new_cell, input_keep_prob=dropout_keep)
        stacked_rnn.append(new_cell)

    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    init_state = multi_cell.zero_state(batch_size, tf.float32)

    rnn_outputs, final_state = tf.nn.dynamic_rnn(multi_cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, nb_class])
        b = tf.get_variable('b', [nb_class], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])
    y_one_hot = tf.one_hot(y_reshaped, nb_class)

    logits = tf.matmul(rnn_outputs, W) + b

    # (batch x num_steps) x nb_class
    predictions = tf.nn.softmax(logits)


    total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot))
    train_step = tf.train.AdamOptimizer().minimize(total_loss)

    predictions_max = tf.argmax(predictions, axis=1)

    correct_prediction = tf.equal(predictions_max, tf.cast(y_reshaped, tf.int64))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    predictions_reshaped = tf.reshape(predictions, [batch_size, num_steps, nb_class])

    tf.summary.scalar("total_loss", total_loss)
    tf.summary.scalar("accuracy", accuracy)
    summaries = tf.summary.merge_all()
    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        accuracy = accuracy,
        train_step = train_step,
        predictions = predictions_reshaped,
        saver = tf.train.Saver(),
        summaries = summaries
    )

def train_network(graph, num_steps = 200, batch_size = 32, checkpoint=None, logs=None):
    tf.set_random_seed(1234)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(logs, graph=sess.graph) # for 0.8
        if checkpoint is not None:
            try:
                graph['saver'].restore(sess, checkpoint)
            except:
                print( 'checkpoint restore not found.' )
                pass

        training_state = None
        for training_iter in range(1, 10000000):
            X, Y = train_set.next(batch_size, num_steps)
            feed_dict={graph['x']: X, graph['y']: Y}

            if training_state is not None:
                feed_dict[graph['init_state']] = training_state

            training_state, _ = sess.run([
                                        graph['final_state'],
                                        graph['train_step']],
                                        feed_dict)
            if training_iter % 50 == 0 :
                summary, loss, accuracy = sess.run([graph['summaries'],
                                                    graph['total_loss'],
                                                    graph['accuracy']],
                                                    feed_dict)

                train_writer.add_summary(summary, training_iter)
                graph['saver'].save(sess, checkpoint)
                print("iteration: ", training_iter, ", loss: ", loss, ", accuracy: ", accuracy )

def generate(graph, num_steps = 200, batch_size = 32, checkpoint=None, generation_len=100):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        graph['saver'].restore(sess, checkpoint)
        state = None
        int_to_char = dict((i, c) for i, c in enumerate(train_set.chars))
        X, Y = train_set.next(batch_size, num_steps)
        n_classes = train_set.n_classes
        seed = X[0]

        txt = ''.join([ int_to_char[elt] for elt in seed ])
        txt+= '|'
        for i in range(generation_len):
            if state is not None:
                feed_dict={graph['x']: X, graph['init_state']: state}
            else:
                feed_dict={graph['x']: X }
            preds, state = sess.run([graph['predictions'], graph['final_state']],feed_dict=feed_dict)

            first_batch = preds[0][-1]
            pred = np.random.choice(n_classes, 1, p=first_batch)

            nx = X[0][1:]
            nx = np.append(nx, pred)
            batch_rest = X[1:]
            X = np.append([nx], batch_rest, axis=0)
            txt += int_to_char[pred[0]]
            print(txt)

def test(graph, num_steps = 200, batch_size = 32, checkpoint=None ):
    pass


mode = ['train', 'test', 'generate'][0]

model_hash = 'acc3'

batch_size = 32
num_steps = 10
state_size = 100
num_layers = 2
dropout_keep = 1.0

dataset_path='./data/1984.txt'
model_name = model_hash+'-'+str(num_layers)+'l-'+str(state_size)+'s-'+str(batch_size)+'bs'
checkpoint = './checkpoints/'+model_name+'.ckpt'
logs = './logs/'+model_name

if mode == 'generate':
    batch_size = 1
    dropout_keep = 1
    num_steps = 1

train_set = data_generator(dataset_path, batch_size=batch_size, num_steps=num_steps)

graph = build_graph(
    state_size = state_size,
    nb_class = train_set.n_classes,
    batch_size = batch_size,
    num_steps = num_steps,
    dropout_keep = dropout_keep,
    num_layers = num_layers)

if mode == 'train':
    train_network(graph, num_steps=num_steps, batch_size=batch_size, checkpoint=checkpoint, logs=logs)
elif mode == 'generate':
    generate(graph, num_steps=num_steps, batch_size=batch_size, checkpoint=checkpoint, generation_len=1000)
elif mode == 'test':
    test(graph, num_steps=num_steps, batch_size=batch_size, checkpoint=checkpoint)
