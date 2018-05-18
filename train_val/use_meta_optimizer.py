import CustomOp as co
import RL_farmwork.experience_replay as rl
from models.official.resnet.cifar10_main import *
from models.official.resnet.cifar10_main import _WEIGHT_DECAY, _NUM_IMAGES, _MOMENTUM
from train_val.problems import *


def cifar10_model_fn(features, labels, mode, params):
    """Model function for CIFAR-10."""
    tf.summary.image('images', features, max_outputs=6)

    inputs = features
    network = get_problem(params)
    with tf.variable_scope('nn', reuse=False):
        if params['optimizer'] == 'kfac':
            with kfac_layer_collection() as lc:
                logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)
        else:
            logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Scale the learning rate linearly with the batch size. When the batch size
        # is 128, the learning rate should be 0.1.
        initial_learning_rate = params['lr']  # 0.1 * params['batch_size'] / 128
        # batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
        global_step = tf.train.get_or_create_global_step()

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        # boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
        # values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
        # learning_rate = tf.train.piecewise_constant(
        #    tf.cast(global_step, tf.int32), boundaries, values)
        learning_rate = initial_learning_rate

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        if params['optimizer'] == 'meta':
            optimizer = co.MetaHessionFreeOptimizer(learning_rate=learning_rate,
                                                    iter=params['CG_iter'],
                                                    x_use=params['x_use'],
                                                    y_use=params['y_use'])
        elif params['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif params['optimizer'] == 'RMSprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif params['optimizer'] == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=_MOMENTUM)
        elif params['optimizer'] == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif params['optimizer'] == 'kfac':
            optimizer = tfcb.kfac.optimizer.KfacOptimizer(learning_rate=learning_rate,
                                                          cov_ema_decay=0.9,
                                                          damping=1e-3,
                                                          layer_collection=lc.layer_collection)
        else:
            raise ValueError


        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if params['optimizer'] == 'meta':
                train_op = optimizer.minimize(loss_type='cross_entropy',
                                              out=logits,
                                              label=labels,
                                              input_list=[inputs],
                                              global_step=global_step)
                train_hooks = [co.MetaParametersLoadingHook(params['meta_ckpt'])]
            else:
                train_op = optimizer.minimize(loss, global_step=global_step)
                train_hooks = [rl.RecordStateHook(state_scope='nn',
                                                  total_step=total_step,
                                                  account=100,
                                                  loss=cross_entropy,
                                                  experience=experience)]
    else:
        train_op = None
        train_hooks = []

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        training_hooks=train_hooks)


def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Set up a RunConfig to only save checkpoints once per training cycle.
    Session_cfg = dict({
        'log_device_placement': False,
        'gpu_options': tf.GPUOptions(
            allow_growth=True,
        ),
        'allow_soft_placement': True,
    })

    config = tf.ConfigProto(**Session_cfg)
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                                  session_config=config)
    cifar_classifier = tf.estimator.Estimator(
        model_fn=cifar10_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'resnet_size': FLAGS.resnet_size,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'meta_ckpt': FLAGS.meta_ckpt,
            'optimizer': FLAGS.optimizer,
            'lr': FLAGS.lr,
            'CG_iter': FLAGS.CG_iter,
            'x_use': FLAGS.x_use,
            'y_use': FLAGS.y_use,
            'problem': FLAGS.problem
        })

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        cifar_classifier.train(
            input_fn=lambda: input_fn(
                True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = cifar_classifier.evaluate(
            input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
        print(eval_results)


parser.add_argument('--meta_ckpt', type=str, default='/tmp/cifar10_data',
                    help='The path to the metackpt_data')

parser.add_argument('--lr', type=float, default=1,
                    help='init lr.')

parser.add_argument('--optimizer', type=str, default='meta',
                    help='chosen of optimizer, ["meta", "SGD", "RMSprop", "adam", "kfac", "momentum"]')

parser.add_argument('--problem', type=str, default='',
                    help='["resnet", "convnet"]')

parser.add_argument('--CG_iter', type=int, default=2,
                    help='CG iterations.')

parser.add_argument('--x_use', type=str, default='x',
                    help="['x', 'd', 'rnn'].")

parser.add_argument('--y_use', type=str, default='rnn',
                    help="['rnn', 'none']")

if __name__ == '__main__':
    with rl.Manager() as manager:
        tf.logging.set_verbosity(tf.logging.INFO)
        FLAGS, unparsed = parser.parse_known_args()
        total_step = int(_NUM_IMAGES['train'] * FLAGS.train_epochs / FLAGS.batch_size)
        conf = {'size': 100,
                'learn_start': 1,
                'partition_num': 10,
                'total_step': total_step,
                'batch_size': 1}
        experience = manager.Experience(conf, FLAGS.model_dir)
        tf.app.run(argv=[sys.argv[0]] + unparsed)
