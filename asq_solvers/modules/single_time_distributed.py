import torch


class SingleTimeDistributed(torch.nn.Module):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``Module`` that takes
    inputs like ``(batch_size, [rest])``, ``SingleTimeDistributed`` reshapes the tensor in the
    distribute_input position of the input arguments to be ``(batch_size * time_steps, [rest])``,
    applies the contained ``Module``, then reshapes it back.

    The key difference from TimeDistributed is that it will only perform the distribution of
    the second dimension on the tensor in the ``distribute_input`` position in the forward call
    to produce the tensor of size: ``(batch_size * time_steps, [rest])``. For all the other
    tensors shaped like ``(batch_size, [rest])``, it will repeat the tensor to match the
    dimension of the tensor in the ``distribute_input`` position, i.e. ``(batch_size * time_steps,
    [rest])``.
    """
    def __init__(self, distributed_module, distribute_input):
        super(SingleTimeDistributed, self).__init__()
        self._module = distributed_module
        self._distribute_input = distribute_input

    def forward(self, *inputs):
        reshaped_inputs = []
        input_index = 0
        # second dimension of the selected input tensor used for distributing the tensors
        squashed_steps = inputs[self._distribute_input].size()[1]
        for input_tensor in inputs:
            if input_index == self._distribute_input:
                input_size = input_tensor.size()
                if len(input_size) <= 2:
                    raise RuntimeError("No dimension to distribute: " + str(input_size))

                # Squash batch_size and time_steps into a single axis; result has shape
                # (batch_size * time_steps, input_size).
                squashed_shape = [-1] + [x for x in input_size[2:]]
                reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))
            else:
                # For others, repeat the tensor for the squashed time steps to ensure the same
                # dimensionality for the new "batch" dimension: batch_size * time_steps
                input_size = input_tensor.size()
                # first shape into batch x squashed_steps x ...
                new_shape = [input_size[0], squashed_steps] + [x for x in input_size[1:]]
                expanded_tensor = input_tensor.unsqueeze(1).expand(*new_shape)
                # re-shape to the (batch_size * time_steps, input_size)
                squashed_shape = [-1] + [x for x in new_shape[2:]]
                reshaped_inputs.append(expanded_tensor.contiguous().view(*squashed_shape))
            input_index += 1

        reshaped_outputs = self._module(*reshaped_inputs)

        # Now get the output back into the right shape.
        # (batch_size, time_steps, [hidden_size])
        new_shape = [-1, squashed_steps] + [x for x in reshaped_outputs.size()[1:]]
        outputs = reshaped_outputs.contiguous().view(*new_shape)
        return outputs
