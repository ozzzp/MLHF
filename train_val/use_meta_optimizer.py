import kfac

import CustomOp as co
import RL_farmwork.experience_replay as rl
from models.official.resnet.cifar10_main import *
from models.official.resnet.cifar10_main import input_fn, _WEIGHT_DECAY, _NUM_IMAGES
from train_val.problems import *


def get_filenames(training_type, data_dir):
    """Returns a list of filenames."""
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    assert os.path.exists(data_dir), (
        'Run cifar10_download_and_extract.py first to download and extract the '
        'CIFAR-10 data.')

    if training_type == 'meta_training':
        return [
            os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in range(1, 4)
        ]
    elif training_type == 'training':
        return [
            os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in range(4, 6)
        ]
    elif training_type == 'testing':
        return [os.path.join(data_dir, 'test_batch.bin')]


def input_fn(training_type, data_dir, batch_size, num_epochs=1):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.

    Returns:
      A tuple of images and labels.
    """
    dataset = record_dataset(get_filenames(training_type, data_dir))

    is_training = training_type != 'testing'

    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance. Because CIFAR-10
        # is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

    dataset = dataset.map(parse_record, num_parallel_calls=10)
    dataset = dataset.map(
        lambda image, label: (preprocess_image(image, is_training), label), num_parallel_calls=10)

    dataset = dataset.prefetch(2 * batch_size)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)

    # Batch results by up to batch_size, and then fetch the tuple from the
    # iterator.
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels


def cifar10_model_fn(features, labels, mode, params):
    """Model function for CIFAR-10."""
    tf.summary.image('images', features, max_outputs=6)

    inputs = features
    _network = get_problem(params)

    def network(*inputs):
        with tf.variable_scope('nn', reuse=tf.AUTO_REUSE):
            return _network(*inputs, mode == tf.estimator.ModeKeys.TRAIN)

    logits = network(inputs)

    if params['optimizer'] == 'kfac':
        lc = kfac.LayerCollection()
        lc.register_categorical_predictive_distribution(logits)
        lc.auto_register_layers()

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
                                                    y_use=params['y_use'],
                                                    d_use=params['d_use'],
                                                    damping_type=params['damping_type'],
                                                    damping=params['damping'],
                                                    decay=params['decay'])
        elif params['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=params['beta1'],
                                               beta2=params['beta2'])
        elif params['optimizer'] == 'RMSprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=params['decay'])
        elif params['optimizer'] == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=params['momentum'])
        elif params['optimizer'] == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif params['optimizer'] == 'kfac':
            optimizer = kfac.PeriodicInvCovUpdateKfacOpt(learning_rate=learning_rate,
                                                         cov_ema_decay=params['decay'],
                                                         damping=params['damping'],
                                                         layer_collection=lc)

            if params['damping_type'] == 'LM_heuristics':
                last_inputs = tf.get_variable('last_input', initializer=tf.zeros_initializer, shape=inputs.shape,
                                              dtype=inputs.dtype, trainable=False)

                last_labels = tf.get_variable('last_label', initializer=tf.zeros_initializer, shape=labels.shape,
                                              dtype=labels.dtype, trainable=False)

                catched_collecctions = [tf.assign(last_inputs, inputs), tf.assign(last_labels, labels)]

                optimizer.set_damping_adaptation_params(
                    prev_train_batch=(last_inputs, last_labels),
                    is_chief=True,
                    loss_fn=lambda x: tf.losses.softmax_cross_entropy(logits=network(x[0]),
                                                                      onehot_labels=x[1]),
                    damping_adaptation_decay=params['momentum'],
                )
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
                                              global_step=global_step,
                                              network_fn=network)
                train_hooks = [co.MetaParametersLoadingHook(params['meta_ckpt'])]
            else:
                train_op = optimizer.minimize(loss, global_step=global_step)
                '''
                train_hooks = [rl.RecordStateHook(state_scope='nn',
                                                  total_step=total_step,
                                                  account=100,
                                                  loss=cross_entropy,
                                                  experience=experience)]
                '''

                if params['optimizer'] == 'kfac' and params['damping_type'] == 'LM_heuristics':
                    with tf.control_dependencies([train_op]):
                        with tf.control_dependencies(catched_collecctions):
                            train_op = tf.no_op()
                train_hooks = []
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
            'd_use': FLAGS.d_use,
            'problem': FLAGS.problem,
            'damping': FLAGS.damping,
            'beta1': FLAGS.beta1,
            'beta2': FLAGS.beta2,
            'decay': FLAGS.decay,
            'momentum': FLAGS.momentum,
            'damping_type': FLAGS.damping_type
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
                'training', FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = cifar_classifier.evaluate(
            input_fn=lambda: input_fn('testing', FLAGS.data_dir, FLAGS.batch_size))
        print(eval_results)


parser.add_argument('--momentum', type=float, default=0.9,
                    help='')

parser.add_argument('--beta1', type=float, default=0.9,
                    help='')

parser.add_argument('--beta2', type=float, default=0.999,
                    help='')

parser.add_argument('--decay', type=float, default=0.9,
                    help='')

parser.add_argument('--meta_ckpt', type=str, default='/tmp/cifar10_data',
                    help='The path to the metackpt_data')

parser.add_argument('--damping_type', type=str, default='regular',
                    help="['regular', 'LM_heuristics']")

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

parser.add_argument('--d_use', type=str, default='rnn',
                    help="['rnn', 'none']")

parser.add_argument('--damping', type=float, default=2e-5,
                    help="damping")

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
