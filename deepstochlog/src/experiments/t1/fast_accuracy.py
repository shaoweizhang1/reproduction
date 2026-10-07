"""Exact T1 accuracy for the MNIST addition grammar; no upstream counterpart.

Built from the grammar in examples/addition: digit probabilities are combined
through a carry DP, giving the engine's generation-output distribution without
visiting each candidate's proof tree. Generation-output accuracy only.
"""

import torch
import torch.nn.functional as F
from deepstochlog.term import Term


def _constants(device, dtype):
    digits = torch.arange(10, device=device)
    digit_sums = (digits[:, None] + digits[None, :]).reshape(-1)
    sum_matrix = F.one_hot(digit_sums, num_classes=19).to(dtype=dtype)

    carry_in = torch.arange(2, device=device)[:, None, None]
    output_digit = digits[None, :, None]
    carry_out = torch.arange(2, device=device)[None, None, :]
    pair_index = output_digit + 10 * carry_out - carry_in
    valid = (pair_index >= 0) & (pair_index < 19)
    return sum_matrix, pair_index.clamp(0, 18), valid


def _sum_distribution(digit_probabilities, sum_matrix, pair_index, valid):
    batch_size, twice_length, classes = digit_probabilities.shape
    assert classes == 10 and twice_length % 2 == 0
    length = twice_length // 2

    state = digit_probabilities.new_zeros((batch_size, 1, 2))
    state[:, 0, 0] = 1.0
    for position in range(length - 1, -1, -1):
        left = digit_probabilities[:, position]
        right = digit_probabilities[:, length + position]
        pair = (left[:, :, None] * right[:, None, :]).reshape(batch_size, 100)
        pair_sums = pair @ sum_matrix
        transition = pair_sums[:, pair_index] * valid
        next_state = torch.einsum("bsc,bcyo->bsyo", state, transition)
        state = next_state.permute(0, 2, 1, 3).reshape(batch_size, -1, 2)

    return torch.cat((state[:, :, 0], state[:, :, 1]), dim=1)


def calculate_accuracy(
    model, test_data, generation_output_accuracy=True,
    most_probable_parse_accuracy=False, create_parse=None, **kwargs,
):
    if not generation_output_accuracy or most_probable_parse_accuracy:
        raise NotImplementedError("Fast T1 accuracy supports generation output only")
    if len(test_data) == 0:
        return 0, 0, 0

    network = model.neural_networks.networks["number"].neural_model
    device = next(network.parameters()).device
    total, correct, label_probability = 0, 0, 0.0
    constants = None

    with torch.no_grad():
        for batch in test_data:
            if not batch:
                continue
            length = int(batch[0].term.arguments[1].functor)
            assert all(int(item.term.arguments[1].functor) == length for item in batch)
            images = torch.stack([
                item.context.get_tensor_representation(Term("t%d" % (i + 1)))
                for item in batch for i in range(2 * length)
            ]).to(device)
            digit_probabilities = network(images).reshape(len(batch), 2 * length, 10)
            if constants is None:
                constants = _constants(device, digit_probabilities.dtype)
            distribution = _sum_distribution(digit_probabilities, *constants)
            targets = torch.tensor(
                [int(item.term.arguments[0].functor) for item in batch],
                device=device, dtype=torch.long,
            )
            own = distribution.gather(1, targets[:, None]).squeeze(1)
            correct += int((own >= distribution.max(dim=1).values).sum().item())
            label_probability += float(own.sum().item())
            total += len(batch)

    if total == 0:
        return 0, 0, 0
    return correct / total, label_probability / total, 0
