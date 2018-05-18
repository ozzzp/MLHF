from tensorflow.python.ops.gradients_impl import *
from tensorflow.python.ops.gradients_impl import _DefaultGradYs, \
    _AsList, _PendingCount, _SetGrad, _IsTrainable, _MaybeCompile, \
    _StopOps, _UpdatePendingAndEnqueueReady, _GetGrad, _LogOpGradients, \
    _VerifyGeneratedGradients, _AggregatedGrads, _maybe_colocate_with, _SymGrad


def gradients(ys,
              xs,
              grad_ys=None,
              name="gradients",
              colocate_gradients_with_ops=False,
              gate_gradients=False,
              aggregation_method=None,
              stop_gradients=None):
    """Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`.

    `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
    is a list of `Tensor`, holding the gradients received by the
    `ys`. The list must be the same length as `ys`.

    `gradients()` adds ops to the graph to output the derivatives of `ys` with
    respect to `xs`.  It returns a list of `Tensor` of length `len(xs)` where
    each tensor is the `sum(dy/dx)` for y in `ys`.

    `grad_ys` is a list of tensors of the same length as `ys` that holds
    the initial gradients for each y in `ys`.  When `grad_ys` is None,
    we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
    user can provide their own initial `grad_ys` to compute the
    derivatives using a different initial gradient for each y (e.g., if
    one wanted to weight the gradient differently for each value in
    each y).

    `stop_gradients` is a `Tensor` or a list of tensors to be considered constant
    with respect to all `xs`. These tensors will not be backpropagated through,
    as though they had been explicitly disconnected using `stop_gradient`.  Among
    other things, this allows computation of partial derivatives as opposed to
    total derivatives. For example:

    ```python
    a = tf.constant(0.)
    b = 2 * a
    g = tf.gradients(a + b, [a, b], stop_gradients=[a, b])
    ```

    Here the partial derivatives `g` evaluate to `[1.0, 1.0]`, compared to the
    total derivatives `tf.gradients(a + b, [a, b])`, which take into account the
    influence of `a` on `b` and evaluate to `[3.0, 1.0]`.  Note that the above is
    equivalent to:

    ```python
    a = tf.stop_gradient(tf.constant(0.))
    b = tf.stop_gradient(2 * a)
    g = tf.gradients(a + b, [a, b])
    ```

    `stop_gradients` provides a way of stopping gradient after the graph has
    already been constructed, as compared to `tf.stop_gradient` which is used
    during graph construction.  When the two approaches are combined,
    backpropagation stops at both `tf.stop_gradient` nodes and nodes in
    `stop_gradients`, whichever is encountered first.

    Args:
      ys: A `Tensor` or list of tensors to be differentiated.
      xs: A `Tensor` or list of tensors to be used for differentiation.
      grad_ys: Optional. A `Tensor` or list of tensors the same size as
        `ys` and holding the gradients computed for each y in `ys`.
      name: Optional name to use for grouping all the gradient ops together.
        defaults to 'gradients'.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      gate_gradients: If True, add a tuple around the gradients returned
        for an operations.  This avoids some race conditions.
      aggregation_method: Specifies the method used to combine gradient terms.
        Accepted values are constants defined in the class `AggregationMethod`.
      stop_gradients: Optional. A `Tensor` or list of tensors not to differentiate
        through.

    Returns:
      A list of `sum(dy/dx)` for each x in `xs`.

    Raises:
      LookupError: if one of the operations between `x` and `y` does not
        have a registered gradient function.
      ValueError: if the arguments are invalid.
      RuntimeError: if called in Eager mode.

    """
    # Creating the gradient graph for control flow mutates Operations. _lock
    # ensures a Session.run call cannot occur between creating and mutating new
    # ops.
    with ops.get_default_graph()._lock:  # pylint: disable=protected-access
        return _GradientsHelper(ys, xs, grad_ys, name, colocate_gradients_with_ops,
                                gate_gradients, aggregation_method, stop_gradients)


def _GradientsHelper(ys, xs, grad_ys, name, colocate_gradients_with_ops,
                     gate_gradients, aggregation_method, stop_gradients):
    """Implementation of gradients()."""
    if context.executing_eagerly():
        raise RuntimeError("tf.gradients not supported when eager execution "
                           "is enabled. Use tf.contrib.eager.GradientTape "
                           "instead.")
    ys = _AsList(ys)
    xs = _AsList(xs)
    stop_gradients = [] if stop_gradients is None else _AsList(stop_gradients)
    if grad_ys is None:
        grad_ys = [None] * len(ys)
    else:
        grad_ys = _AsList(grad_ys)

    with ops.name_scope(
            name, "gradients",
            list(ys) + list(xs) + list(stop_gradients) + list(grad_ys)) as grad_scope:
        ys = ops.convert_n_to_tensor_or_indexed_slices(ys, name="y")
        xs = [
            x.handle if resource_variable_ops.is_resource_variable(x) else x
            for x in xs
        ]
        xs = ops.internal_convert_n_to_tensor_or_indexed_slices(
            xs, name="x", as_ref=True)
        grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops)

        # The approach we take here is as follows: Create a list of all ops in the
        # subgraph between the ys and xs.  Visit these ops in reverse order of ids
        # to ensure that when we visit an op the gradients w.r.t its outputs have
        # been collected.  Then aggregate these gradients if needed, call the op's
        # gradient function, and add the generated gradients to the gradients for
        # its input.

        # Initialize the pending count for ops in the connected subgraph from ys
        # to the xs.
        if len(ys) > 1:
            ys = [array_ops.identity(y) if y.consumers() else y for y in ys]
        to_ops = [t.op for t in ys]
        from_ops = [t.op for t in xs]
        stop_gradient_ops = [t.op for t in stop_gradients]
        pending_count, loop_state = _PendingCount(
            ys[0].graph, to_ops, from_ops, colocate_gradients_with_ops)

        # Iterate over the collected ops.
        #
        # grads: op => list of gradients received on each output endpoint of the
        # op.  The gradients for each endpoint are initially collected as a list.
        # When it is time to call the op's gradient function, for each endpoint we
        # aggregate the list of received gradients into a Add() Operation if there
        # is more than one.
        grads = {}

        # Add the initial gradients for the ys.
        for y, grad_y in zip(ys, grad_ys):
            _SetGrad(grads, y, grad_y)

        # Initialize queue with to_ops.
        queue = collections.deque()
        # Add the ops in 'to_ops' into the queue.
        to_ops_set = set()
        for op in to_ops:
            # 'ready' handles the case where one output gradient relies on
            # another output's gradient.
            # pylint: disable=protected-access
            ready = (pending_count[op._id] == 0)
            if ready and op._id not in to_ops_set:
                to_ops_set.add(op._id)
                queue.append(op)
            # pylint: enable=protected-access

        if loop_state:
            loop_exits = loop_state.ProcessUnusedLoopExits(pending_count, to_ops_set)
            for y in loop_exits:
                if _IsTrainable(y):
                    _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
                    queue.append(y.op)

        stop_ops = _StopOps(from_ops, stop_gradient_ops, pending_count)
        while queue:
            # generate gradient subgraph for op.
            op = queue.popleft()
            with _maybe_colocate_with(op, colocate_gradients_with_ops):
                if loop_state:
                    loop_state.EnterGradWhileContext(op, before=True)
                out_grads = _AggregatedGrads(grads, op, loop_state, aggregation_method)
                if loop_state:
                    loop_state.ExitGradWhileContext(op, before=True)

                grad_fn = None
                # pylint: disable=protected-access
                func_call = None
                is_func_call = ops.get_default_graph()._is_function(op.type)
                has_out_grads = any(isinstance(g, ops.Tensor) or g for g in out_grads)
                if has_out_grads and (op._id not in stop_ops):
                    if is_func_call:
                        func_call = ops.get_default_graph()._get_function(op.type)
                        grad_fn = func_call.python_grad_func
                        # pylint: enable=protected-access
                    else:
                        # A grad_fn must be defined, either as a function or as None
                        # for ops that do not have gradients.
                        try:
                            grad_fn = ops.get_gradient_function(op)
                        except LookupError:
                            raise LookupError(
                                "No gradient defined for operation '%s' (op type: %s)" %
                                (op.name, op.type))
                if loop_state:
                    loop_state.EnterGradWhileContext(op, before=False)
                if (grad_fn or is_func_call) and has_out_grads:
                    # NOTE: If _AggregatedGrads didn't compute a value for the i'th
                    # output, it means that the cost does not depend on output[i],
                    # therefore dC/doutput[i] is 0.
                    for i, out_grad in enumerate(out_grads):
                        if (not isinstance(out_grad, ops.Tensor) and not out_grad) and (
                                (not grad_fn and is_func_call) or _IsTrainable(op.outputs[i])):
                            # Only trainable outputs or outputs for a function call that
                            # will use SymbolicGradient get a zero gradient. Gradient
                            # functions should ignore the gradient for other outputs.
                            # TODO(apassos) gradients of resource handles might be an
                            # issue here because of zeros.
                            if loop_state:
                                out_grads[i] = loop_state.ZerosLike(op, i)
                            else:
                                out_grads[i] = control_flow_ops.ZerosLikeOutsideLoop(op, i)
                    with ops.name_scope(op.name + "_grad"):
                        # pylint: disable=protected-access
                        with ops.get_default_graph()._original_op(op):
                            # pylint: enable=protected-access
                            if grad_fn:
                                # If grad_fn was found, do not use SymbolicGradient even for
                                # functions.
                                in_grads = _MaybeCompile(grad_scope, op, func_call,
                                                         lambda: grad_fn(op, *out_grads))
                            else:
                                # For function call ops, we add a 'SymbolicGradient'
                                # node to the graph to compute gradients.
                                in_grads = _MaybeCompile(grad_scope, op, func_call,
                                                         lambda: _SymGrad(op, out_grads))
                            in_grads = _AsList(in_grads)
                            _VerifyGeneratedGradients(in_grads, op)
                            if gate_gradients and len([x for x in in_grads
                                                       if x is not None]) > 1:
                                with ops.device(None):
                                    with ops.colocate_with(None, ignore_existing=True):
                                        in_grads = control_flow_ops.tuple(in_grads)
                    _LogOpGradients(op, out_grads, in_grads)
                else:
                    # If no grad_fn is defined or none of out_grads is available,
                    # just propagate a list of None backwards.
                    in_grads = [None] * len(op.inputs)
                for i, (t_in, in_grad) in enumerate(zip(op.inputs, in_grads)):
                    if in_grad is not None:
                        if (isinstance(in_grad, ops.Tensor) and
                                t_in.dtype != dtypes.resource):
                            try:
                                in_grad.set_shape(t_in.get_shape())
                            except ValueError:
                                raise ValueError(
                                    "Incompatible shapes between op input and calculated "
                                    "input gradient.  Forward operation: %s.  Input index: %d. "
                                    "Original input shape: %s.  "
                                    "Calculated input gradient shape: %s" %
                                    (op.name, i, t_in.shape, in_grad.shape))
                        _SetGrad(grads, t_in, in_grad)
                if loop_state:
                    loop_state.ExitGradWhileContext(op, before=False)

            # Update pending count for the inputs of op and enqueue ready ops.
            assert op.graph == ys[0].graph
            _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state)

    if loop_state:
        loop_state.PostProcessing()
    return [_GetGrad(grads, x) for x in xs]


@ops.RegisterGradient("Enter_custom")
def costum_EnterGrad(op, grad):
    """Gradients for an Enter are calculated using an Exit op.

    For loop variables, grad is the gradient so just add an exit.
    For loop invariants, we need to add an accumulator loop.
    """
    graph = op.graph
    # pylint: disable=protected-access
    grad_ctxt = graph._get_control_flow_context()
    # pylint: enable=protected-access
    if not grad_ctxt.back_prop:
        # Skip gradient computation, if the attribute `back_prop` is false.
        return grad
    if grad_ctxt.grad_state is None:
        # Pass the gradient through if we are not in a gradient while context.
        return grad
    if op.get_attr("is_constant"):
        # Add a gradient accumulator for each loop invariant.
        if isinstance(grad, ops.Tensor):
            result = grad_ctxt.AddBackpropAccumulator(op, grad)
        elif isinstance(grad, ops.IndexedSlices):
            result = grad_ctxt.AddBackpropIndexedSlicesAccumulator(op, grad)
        else:
            # TODO(yuanbyu, lukasr): Add support for SparseTensor.
            raise TypeError("Type %s not supported" % type(grad))
    else:
        result = exit(grad)
        grad_ctxt.loop_exits.append(result)
        grad_ctxt.ExitResult([result])
    return result
