import CustomOp as co
import RL_farmwork.experience_replay as rl
from models.official.resnet.cifar10_main import *
from models.official.resnet.cifar10_main import _NUM_IMAGES
from train_val.problems import get_problem


def input_fn(is_training, data_dir, batch_size, meta_roll_back, num_epochs=1):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.

    Returns:
      A tuple of images and labels.
    """
    dataset = record_dataset(get_filenames(is_training, data_dir))

    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance. Because CIFAR-10
        # is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=5000)

    dataset = dataset.map(parse_record, num_parallel_calls=5)
    dataset = dataset.map(
        lambda image, label: (preprocess_image(image, is_training), label), num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)

    # Batch results by up to batch_size, and then fetch the tuple from the
    # iterator.
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.batch(meta_roll_back)
    dataset = dataset.prefetch(10)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels


def cifar10_model_meta_train_fn(features, labels, mode, params):
    """Model function for CIFAR-10."""
    assert mode == tf.estimator.ModeKeys.TRAIN

    network = get_problem(params)
    base_loss, meta_loss, hession_loss, r_loss, update_fn = co.build_meta_train_rnn(network, features, labels, params)

    meta_varibles = tf.global_variables(scope='metaoptimizer')
    var_regularation = tf.add_n([tf.nn.l2_loss(v) for v in meta_varibles])

    regular_loss = 2e-7 * var_regularation

    total_loss = hession_loss + meta_loss + regular_loss

    tf.identity(total_loss, name='loss')
    tf.summary.scalar('loss', total_loss)

    initial_learning_rate = params['meta_lr']
    batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size'] / params['meta_roll_back']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in [150, 190, 220]]
    print(boundaries)
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def get_grad_and_vars():
        grad_and_vars = optimizer.compute_gradients(regular_loss, var_list=meta_varibles,
                                                    gate_gradients=optimizer.GATE_GRAPH)
        var_list = tf.global_variables(scope='metaoptimizer/x') + tf.global_variables(scope='metaoptimizer/y')
        if len(var_list) != 0:
            grad_and_vars_part_1 = optimizer.compute_gradients(hession_loss,
                                                               var_list=var_list,
                                                               gate_gradients=optimizer.GATE_GRAPH)
        else:
            grad_and_vars_part_1 = []

        var_list = tf.global_variables(scope='metaoptimizer/d')
        if len(var_list) != 0:
            grad_and_vars_part_2 = optimizer.compute_gradients(10 * meta_loss,
                                                               var_list=var_list,
                                                               gate_gradients=optimizer.GATE_GRAPH)
        else:
            grad_and_vars_part_2 = []

        return co.merge_grad_and_vars(grad_and_vars, grad_and_vars_part_1, grad_and_vars_part_2)

    if params['debug_grad']:
        with co.debug_grad():
            grad_and_vars = get_grad_and_vars()
    else:
        grad_and_vars = get_grad_and_vars()

    var_length = tf.global_norm(meta_varibles)
    tf.identity(var_length, name='var_length')
    tf.summary.scalar('var_length', var_length)

    grads, grad_length = tf.clip_by_global_norm([g for g, _ in grad_and_vars], var_length)
    grad_and_vars = [(g, v) for g, (_, v) in zip(grads, grad_and_vars)]

    grad_length = tf.minimum(grad_length, var_length) * learning_rate
    tf.identity(grad_length, name='grad_length')
    tf.summary.scalar('grad_length', grad_length)

    grad_rate = grad_length / var_length
    tf.identity(grad_rate, name='grad_rate')
    tf.summary.scalar('grad_rate', grad_rate)

    with tf.control_dependencies([tf.verify_tensor_all_finite(grad_length, 'meta gradient NaN')]):
        meta_train_op = optimizer.apply_gradients(grad_and_vars, global_step=global_step)
    with tf.control_dependencies([meta_train_op]):
        with tf.control_dependencies([update_fn()]):
            train_op = tf.no_op()

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        training_hooks=[rl.SaveStateHook(state_scope='nn',
                                         reset_scope='slots',
                                         meta_error=meta_loss,
                                         base_error=base_loss,
                                         keep_prob=params['keep_prob'], experience=experience)])

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
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=1000,
                                                  save_summary_steps=10,
                                                  session_config=config)
    cifar_classifier = tf.estimator.Estimator(
        model_fn=cifar10_model_meta_train_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'resnet_size': FLAGS.resnet_size,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'meta_roll_back': FLAGS.meta_roll_back,
            'meta_lr': FLAGS.meta_lr,
            'keep_prob': FLAGS.keep_prob,
            'lr': FLAGS.lr,
            'CG_iter': FLAGS.CG_iter,
            'print_log_in_meta_optimizer': FLAGS.print_log_in_meta_optimizer,
            'x_use': FLAGS.x_use,
            'y_use': FLAGS.y_use,
            'problem': FLAGS.problem,
            'debug_grad': FLAGS.debug_grad
        })

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'hession_loss': 'hession_loss',
            'meta_loss': 'meta_loss',
            'base_loss': 'base_loss',
            'loss': 'loss',
            'var_length': 'var_length',
            'grad_length': 'grad_length',
            'grad_rate': 'grad_rate',
            'r_loss': 'r_loss'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)

        cifar_classifier.train(
            input_fn=lambda: input_fn(
                True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.meta_roll_back, FLAGS.epochs_per_eval),
            hooks=[logging_hook])


parser.add_argument('--meta_roll_back', type=int, default=10,
                    help='rollback of meta learning.')

parser.add_argument('--keep_prob', type=float, default=0.5,
                    help='keep_prob of experience replay.')


parser.add_argument('--problem', type=str, default='',
                    help='["resnet", "convnet"]')

parser.add_argument('--meta_lr', type=float, default=0.001,
                    help='init meta lr.')

parser.add_argument('--lr', type=float, default=1.0,
                    help='init lr.')

parser.add_argument('--CG_iter', type=int, default=2,
                    help='CG iterations.')

parser.add_argument('--print_log_in_meta_optimizer', type=bool, default=False,
                    help='print log in meta optimizer.')

parser.add_argument('--x_use', type=str, default='x',
                    help="['x', 'd', 'rnn'].")

parser.add_argument('--y_use', type=str, default='rnn',
                    help="['rnn', 'none']")

parser.add_argument('--debug_grad', type=bool, default=False,
                    help="if debug grad")

# one 8G GPU, e.g. 1080, recommends: batch_size = 64, meta_roll_back=20, resnet_size=14, lr=0.001
# one 12G GPU, e.g. Titan Xp, recommends: batch_size = 64, meta_roll_back=20, resnet_size=32, lr=0.001

if __name__ == '__main__':
    with rl.Manager() as manager:
        tf.logging.set_verbosity(tf.logging.INFO)
        FLAGS, unparsed = parser.parse_known_args()
        conf = {'size': 200,
                'learn_start': 1,
                'partition_num': 10,
                'total_step': int(_NUM_IMAGES['train'] * FLAGS.train_epochs / FLAGS.batch_size / FLAGS.meta_roll_back),
                'batch_size': 1}
        experience = manager.Experience(conf, FLAGS.model_dir)
        tf.app.run(argv=[sys.argv[0]] + unparsed)
