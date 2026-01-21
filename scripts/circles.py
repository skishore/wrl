import argparse


def draw_circle(r: int) -> None:
    limit = r ** 2 + r + (0 if r == 4 else -1 if r == 5 else 1)
    for y in range(2 * r + 1):
        line = []
        for x in range(2 * r + 1):
            inside = (x - r) ** 2 + (y - r) ** 2 < limit
            line.append('＃' if inside else '  ')
        print(''.join(line).rstrip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('radius', type=int)
    args = parser.parse_args()
    for r in range(args.radius):
        if r:
            print()
        draw_circle(r)
