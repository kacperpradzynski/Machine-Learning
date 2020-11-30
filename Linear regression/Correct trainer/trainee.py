import sys
import argparse


def main(args):
    description = []
    input = []
    k = 0

    description_file = open(args['description'], 'r') 
    lines = description_file.readlines() 
    for line in lines: 
        description.append(map(float, line.split()))
    
    for line in sys.stdin:
        input.append(map(float, line.split()))

    k = int(description[0][1])
    description.pop(0)

    for row in input:
        result = 0
        for part in description:
            tmp = 1
            for degree in range(k):
                index = int(part[degree])
                if index != 0:
                    tmp *= row[index - 1]
            result += tmp * part[-1]
        print(result)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--description", type=str, required=True, help="Description file")
    args = vars(ap.parse_args())
    main(args)
