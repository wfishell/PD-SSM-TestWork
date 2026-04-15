import numpy as np

modulus = 5
vocab_size = modulus + 5  # = 10
# Tokens: 1-3=operators (+,-,*), 4='=', 5-9=numbers (value = token - 5)
# Labels: 5..9 (result mod 5, shifted by 5)


def generate_sample(min_length, max_length, rng):
    """
    rng: np.random.Generator
    returns: (tokens: np.ndarray int32, label: int)

    Sequence format: num op num op ... num =
    Length is always even (padded up by 1 if odd input length drawn).
    Evaluation follows BIDMAS: multiplication before addition/subtraction.
    """
    original_length = int(rng.integers(min_length, max_length + 1))
    length = original_length if original_length % 2 == 0 else original_length + 1

    res = [0] * length

    for i in range(0, length, 2):
        res[i] = int(rng.integers(5, 10))      # numbers: tokens 5-9

    for i in range(1, length - 1, 2):
        res[i] = int(rng.integers(1, 4))        # operators: 1=+, 2=-, 3=*

    res[-1] = 4                                  # '=' token

    # Evaluate: multiplication first, then addition/subtraction
    values = [res[0] - 5]
    operators = []
    for i in range(1, length - 2, 2):
        op = res[i]
        num = res[i + 1] - 5
        if op == 3:
            values[-1] *= num
        else:
            values.append(num)
            operators.append(op)

    total = values[0]
    for j, op in enumerate(operators):
        if op == 1:
            total += values[j + 1]
        elif op == 2:
            total -= values[j + 1]

    label = int(total % modulus + 5)
    return np.array(res, dtype=np.int32), label
