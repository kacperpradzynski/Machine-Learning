import numpy
from random import randint

data_in_file = open("task.csv", 'r')

data_in = []
for line in data_in_file.readlines() : 
        data_in.append(line.split(';'))

for data in data_in:
    value = randint(0, 5)
    data[-1] = str(value)

a = numpy.asarray(data_in)
numpy.savetxt("submission.csv", a, delimiter=";",fmt='%s')