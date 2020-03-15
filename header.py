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


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "-r", "--range", required=True,
        help="range of header"
    )

    args = vars(args.parse_args())

    file_name = "header.arff"
    with open(file_name, 'w') as text:
        text.write(make_header(int(args["range"])))
