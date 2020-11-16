import sys

numbers = []
for line in sys.stdin:
    numbers.append(int(line))

numbers.sort()
print(*numbers, sep='\n')