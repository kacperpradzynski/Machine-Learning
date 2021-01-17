import pandas as pd
import numpy as np
from scipy import spatial
import operator
import os.path

def getVector(user):
    vector = [-10] * 200
    train_user = train.loc[train['userid'] == user]
    for index, row in train_user.iterrows():
        vector[int(row['movieid']) - 1] = int(row['rate'])

    return vector

def getNeighborsInCorrectOrder(current_user, users):
    current_user_vector = np.array(getVector(current_user))
    distances = []

    for user in users:
        if user == current_user:
            continue
        user_vector = np.array(getVector(user))
        current_user_vector_shrink = []
        user_vector_shrink = []
        for index in np.where(current_user_vector + user_vector >= 0)[0]:
            current_user_vector_shrink.append(current_user_vector[int(index)])
            user_vector_shrink.append(user_vector[int(index)])
        dist = spatial.distance.cosine(current_user_vector_shrink, user_vector_shrink)
        distances.append((user, dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for distance in distances:
        neighbors.append(distance[0])

    return [current_user, neighbors]

def common_elements(list1, list2):
    result = []
    for element in list1:
        if element in list2:
            result.append(element)
    return result

train = pd.read_csv('./train.csv', sep=';', names=["id", "userid", "movieid", "rate"])
task = pd.read_csv('./task.csv', sep=';', names=["id", "userid", "movieid", "rate"])

if os.path.isfile('./neighbors.csv'):
    neighbors = pd.read_csv('./neighbors.csv')
    neighbors['neighbors'] = neighbors['neighbors'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
    neighbors['neighbors'] = neighbors['neighbors'].str.split(',')
else:
    neighbors = pd.DataFrame(columns=['user', 'neighbors'])
    users = train.userid.unique()
    counter = 0
    for user in users:
        print("User: " + str(counter + 1) + " / " + str(len(users)), end ='\r')
        neighbors.loc[counter] = getNeighborsInCorrectOrder(user, users)
        counter = counter + 1
    neighbors.to_csv('neighbors.csv', index=False)
    print()

for index, row in task.iterrows():
    print("Task: " + str(index) + " / " + str(len(task.index)), end ='\r')
    users_with_movie = train.loc[train['movieid'] == row["movieid"]].userid.unique()
    users = neighbors.loc[neighbors['user'] == row["userid"]].iloc[0]['neighbors']
    top_users = common_elements(users, users_with_movie)[:5]
    neighbors_rate = train.loc[(train['userid'].isin(top_users)) & (train['movieid'] == row["movieid"])]
    task.loc[index, 'rate'] = str(int(round(neighbors_rate['rate'].mean())))

task.to_csv('submission.csv', sep=';', index=False, header=False)
