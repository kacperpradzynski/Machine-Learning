import pandas as pd
import numpy as np

N = 5
EPOCHS = 500
LEARNING_RATE = 0.5

def calculate_gradient(values, loss, has_value):
    gradient = []
    for i in range(len(values)):
        tmp = []
        for j in range(len(loss[0])):
            value = 0
            iter = 0
            for k in range(len(loss)):
                if has_value[k, j]:
                    value += values[i, k] * loss[k, j]
                    iter += 1
            tmp.append(value / iter)
        gradient.append(np.array(tmp))

    return np.array(gradient)

train = pd.read_csv("./train.csv", sep=';', header=None, names=['id', 'user_id', 'movie_id', 'rating'])
train['rating'] = train['rating'] / 5
task = pd.read_csv("./task.csv", sep=';', header=None, names=['id', 'user_id', 'movie_id', 'rating'])
task = task.set_index('id')

rating_matrix = train.pivot_table(index='movie_id', columns='user_id', values='rating')
expected = np.array(rating_matrix.values)

filled_matrix = rating_matrix.fillna(-1)
has_value = filled_matrix != -1
has_value = np.array(has_value.values)

R = len(train['user_id'].unique())
M = len(train['movie_id'].unique())
inputs = np.random.uniform(0, 1, [M, N + 1])
weights = np.random.uniform(0, 1, [N + 1, R])
for i in range(len(inputs)):
    inputs[i][-1] = 1

for i in range(EPOCHS):
    print(f'Epoch: {i+1}/{EPOCHS}', end='\r')
    outputs = inputs.dot(weights)
    loss = outputs - expected

    inputs = inputs - calculate_gradient(weights, loss.T, has_value.T).T * LEARNING_RATE
    weights = weights - calculate_gradient(inputs.T, loss, has_value) * LEARNING_RATE

    for j in range(len(inputs)):
        inputs[j][-1] = 1

outputs = inputs.dot(weights)
rating_matrix = pd.DataFrame.from_records(outputs, columns=rating_matrix.columns)

for i, row in task.iterrows():
    user = row['user_id']
    movie = row['movie_id']
    rate = round(rating_matrix.at[movie - 1, user] * 5)
    task.at[i, 'rating'] = 5 if rate > 5 else 0 if rate < 0 else rate

task['rating'] = task['rating'].astype(int)
task.to_csv('submission.csv', header=False, sep=';')
