import sys
import argparse


def main(args):
    set_file = open(args['set'], 'r')

    set_list = []
    in_list = []

    for line in set_file.readlines() : 
        set_list.append([float(i) for i in line.split()])

    for line in sys.stdin:
        in_list.append([float(i) for i in line.split()])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--set", type=str, required=True, help="set.txt file")
    args = vars(ap.parse_args())
    main(args)
