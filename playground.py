import sys


def main():
    output = []
    for line in sys.stdin:
        output.append(line.strip())
    print(output)


if __name__ == "__main__":
    main()
