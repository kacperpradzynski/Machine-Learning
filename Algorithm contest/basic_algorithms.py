import sys

numbers = []
for line in sys.stdin:
    numbers.append(int(line))

numbers.sort()
for n in numbers:
    print(n)