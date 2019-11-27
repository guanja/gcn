from __future__ import division
from __future__ import print_function

import time
import ipdb

import tensorflow as tf


from gcn.utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

ipdb.set_trace()

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.compat.v1.placeholder(tf.int32),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function.
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, 
                                        placeholders)
    scores = sess.run([model.loss, model.accuracy, model.recall, 
                         model.precision], feed_dict=feed_dict_val)
    time_elapsed = time.time() - t_test
    return scores[0], scores[1], scores[2], scores[3], time_elapsed


# Init variables
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.local_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, 
                                    placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, 
                     model.recall, model.precision], feed_dict=feed_dict)

    # Validation
    cost, acc, rec, prec, duration = evaluate(features, support, y_val, 
                                              val_mask, placeholders)
    cost_val.append(cost)

    # Print results.
    print(f'Epoch: {epoch+1} (time={time.time() - t})\n',
          f'Training: ',
          f'loss={outs[1]:.5f}, acc={outs[2]:.3f}, rec={outs[3][0]:.3f}, ',
          f'prec={outs[4][0]:.3f}\n',
          f'Validation: ',
          f'loss={cost:.5f}, acc={acc:.3f}, rec={rec[0]:.3f}, ',
          f'prec={prec[0]:.3f}.')

    if epoch > FLAGS.early_stopping and \
        cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_rec, test_prec, test_duration = \
    evaluate(features, support, y_test, test_mask, placeholders)

print(f'Test: ({test_duration})\n',
      f'loss={test_cost:.5f}, acc={test_acc:.3f}, rec={test_rec[0]:.3f}, '\
      f'prec={test_prec[0]:.3f}\n')

