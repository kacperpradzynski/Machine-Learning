import sys
import argparse


def predict(row, coefficients):
    prediction = coefficients[0]
    for i in range(len(row)-1):
        prediction += coefficients[i + 1] * row[i]
    return prediction


def get_coefficients(coef, train_set, learning_rate, iterations):
    iteration = 0
    prev_mse = None
    for epoch in range(iterations):
        summation = 0
        for row in train_set:
            prediction = predict(row, coef)
            error = prediction - row[-1]
            summation = summation + (error * error)
            coef[0] = coef[0] - learning_rate * error
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] - learning_rate * error * row[i]
        mse = summation/len(train_set)
        iteration = iteration + 1
        if prev_mse != None and prev_mse - mse < 0.00001:
            break
        prev_mse = mse
    return coef, iteration


def main(args):
    train_set_file = open(args['trainset'], 'r') 
    data_in_file = open(args['datain'], 'r') 
    data_out_file = open(args['dataout'], 'w+') 

    description_in = []
    description_out = []
    train_set = []
    coefficients = []
    iterations = 0

    for line in sys.stdin:
        description_in.append(map(float, line.split()))
    description_out.append(map(int, description_in.pop(0)))
    n = int(description_out[0][0])
    k = int(description_out[0][1])

    for line in train_set_file.readlines() : 
        train_set.append(map(float, line.split()))
    
    for line in data_in_file.readlines() : 
        variable = line.split('=')
        if variable[0] == 'iterations':
            iterations = int(variable[1])

    for row in description_in:
        coefficients.insert(0, row[1])

    coefficients, iterations = get_coefficients(coefficients, train_set, 0.001, iterations)

    data_out_file.write("iterations=%s" % iterations)
    data_out_file.close()

    for i in range(n, -1, -1):
        description_out.append([i, coefficients[i]])

    for line in description_out:
        print(str(line[0]) + " " + str(line[1]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--trainset", type=str, required=True, help="train_set.txt file")
    ap.add_argument("-i", "--datain", type=str, required=True, help="data_in.txt file")
    ap.add_argument("-o", "--dataout", type=str, required=True, help="data_out.txt file")
    args = vars(ap.parse_args())
    main(args)
