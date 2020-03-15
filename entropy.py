from scipy.io import arff
import pandas as pd
import math
import argparse


def make_header(rangeSize):
    header = "@relation BaseImagens"
    for i in range(rangeSize):
        header += f"\n@attribute h{i} NUMERIC"

    header += "\n@attribute classe {"
    for i in range(1, 9):
        header += f"{i},"

    header += "9}\n@data\n\n"
    return header


def entropy(values, size):
    position = 0
    new_values = []
    while position < 511:
        to_add = 0
        for i in range(size):
            if float(values[position + i]):
                to_add = to_add - \
                    (float(values[position + i]) *
                     math.log(float(values[position + i])))

        position += size
        new_values.append(to_add)
    return new_values


def make_arff_file(values, classes):
    range = len(values[0])
    file_name = f"entropy_size{range}.arff"
    file = open(file_name, 'w')
    file.write(make_header(range))

    for index, value in enumerate(values):
        line = ','.join(map(str, value))
        line = line + ',' + classes[index] + '\n'
        file.write(line)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "-s", "--size", required=True,
        help="divide array lenght by size"
    )
    args = vars(args.parse_args())

    file = open('results.arff', 'r')
    lines = file.readlines()

    start = 517

    values = []
    classes = []

    while start < len(lines):
        text = lines[start].split(',')
        classes.append(text[-1][1])
        text.pop()
        values.append(text)
        start += 1

    new_values = []
    for value in values:
        new_values.append(entropy(value, int(args["size"])))

    make_arff_file(new_values, classes)
