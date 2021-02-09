import sys
import random
import argparse


K_FOLDS = 7	
LEARNING_RATE = 0.1	
N_EPOCHS = 20000	
DEGREES = [2, 3, 4, 5, 6, 7, 8]


def min_max(data):
    min_values = []
    max_values = []

    for i in range(len(data[0])):
        min_values.append(min(data, key=lambda x: x[i])[i])
        max_values.append(max(data, key=lambda x: x[i])[i])

    return min_values, max_values


def normalize(value, min_value, max_value):
    result = (value - min_value) / (max_value - min_value)
    result = (2 * result) - 1
    return result


def denormalize(value, min_value, max_value):
    result = (value + 1) / 2
    result = result * (max_value - min_value) + min_value
    return result


def normalize_data(data, min_values, max_values):
    for i in range(len(data[0])):
        for j in range(len(data)):
            data[j][i] = normalize(data[j][i], min_values[i], max_values[i])

    return data


def generate_folds(data):
    dataset_split = list()
    dataset_copy = list(data)
    fold_size = int(len(data) / K_FOLDS)
    for i in range(K_FOLDS):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def sums(length, total_sum):
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value):
                yield (value,) + permutation


def inputs_permutations(inputs, k):
    n = len(inputs[0])
    perms = []
    for i in range(1, k):
        perms += list(sums(n, i))
    new_inputs = []
    for input in inputs:
        new_input = []
        for perm in perms:
            val = 1
            for i in range(len(perm)):
                if(perm[i] != 0):
                    val *= input[i] ** perm[i]
            new_input.append(val)
        new_inputs.append(new_input)
    return new_inputs


def validate(inputs, expected, coef):
    summation = 0
    for i in range(len(inputs)):
        row = inputs[i]
        exp = expected[i]
        prediction = calculate(row, coef)
        error = prediction - exp
        summation = summation + (error * error)
    mse = summation/len(inputs)

    return mse


def calculate(row, coefficients):
    prediction = coefficients[0]
    for i in range(len(row)-1):
        prediction += coefficients[i + 1] * row[i]
    return prediction


def get_coefficients(inputs, expected, coef):
    for i in range(len(inputs)):
        row = inputs[i]
        exp = expected[i]
        prediction = calculate(row, coef)
        error = prediction - exp
        coef[0] = coef[0] - LEARNING_RATE * error
        for i in range(len(row)-1):
            coef[i + 1] = coef[i + 1] - LEARNING_RATE * error * row[i]

    return coef


def train(training_set, validation_set, degree):
    train_target = []
    train_target_output = []
    validate_target = []
    validate_target_output = []

    for j in training_set.copy():
        train_target_output.append(j[-1])
        train_target.append(j[:-1])

    for j in validation_set.copy():
        validate_target_output.append(j[-1])
        validate_target.append(j[:-1])

    train_data = inputs_permutations(train_target, degree)
    validate_data = inputs_permutations(validate_target, degree)

    weights = [random.uniform(-1, 1) for i in range(len(train_data[0]))]

    validate_error = 0
    error_counter = 0
    last_validate_error = None
    best_validate_error = 0
    best_weights = weights

    for epoch in range(N_EPOCHS):
        weights = get_coefficients(train_data, train_target_output, weights)
        validate_error = validate(validate_data, validate_target_output, weights)

        if last_validate_error is not None and validate_error > last_validate_error:
            error_counter += 1
        else:
            best_validate_error = validate_error
            error_counter = 0
            best_weights = weights

        if error_counter > 500:
            break
        last_validate_error = validate_error

    return best_validate_error, best_weights


def predict(inputs, weights, k):
    n = len(inputs)
    perms = []
    for i in range(1, k):
        perms += list(sums(n, i))

    new_input = []
    for perm in perms:
        val = 1
        for i in range(len(perm)):
            if(perm[i] != 0):
                val *= inputs[i] ** perm[i]
        new_input.append(val)

    return calculate(new_input, weights)


def main(args):
    train_data = []
    test_data = []

    for line in open(args['train_data']):
        train_data.append([float(x) for x in line.split()])

    # for line in open(args['test_data']):
    #     test_data.append([float(x) for x in line.split()])

    for line in sys.stdin:
        test_data.append([float(x) for x in line.split()])

    min_values, max_values = min_max(train_data)

    train_data = normalize_data(train_data, min_values, max_values)
    test_data = normalize_data(test_data, min_values, max_values)

    folds = generate_folds(train_data)

    all_errors = []

    for degree in DEGREES:
        error = 0

        for fold in folds:
            training_set = list(folds.copy())
            training_set.remove(fold)
            training_set = sum(training_set, [])
            validation_set = list()
            for row in fold:
                row_copy = list(row.copy())
                validation_set.append(row_copy)

            fold_error, _ = train(training_set, validation_set, degree)
            error += fold_error

        error /= K_FOLDS
        all_errors.append(error)

    index = all_errors.index(min(all_errors))
    k = DEGREES[int(index)]

    _, best_weights = train(train_data, train_data, k)

    for x in test_data:
        result = predict(x, best_weights, k)
        print(denormalize(result, min_values[-1], max_values[-1]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train_data", type=str, default="set.txt", help="set.txt file")
    ap.add_argument("-e", "--test_data", type=str, default="test.txt", help="test.txt file")
    args = vars(ap.parse_args())
    main(args)
