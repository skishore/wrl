import argparse


def bound(r: int) -> int:
    return r ** 2 + r + (0 if r == 4 else -1 if r == 5 else 1)


def draw_circle(r: int) -> None:
    next_bound = bound(r)
    prev_bound = bound(r - 1) if r else -1
    for y in range(2 * r + 1):
        line = []
        for x in range(2 * r + 1):
            l2_squared = (x - r) ** 2 + (y - r) ** 2
            inside_next = l2_squared < next_bound
            inside_prev = l2_squared < prev_bound
            line.append('Ｘ' if inside_prev else '＃' if inside_next else '  ')
        print(''.join(line).rstrip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('radius', type=int)
    args = parser.parse_args()
    for r in range(args.radius):
        if r:
            print()
        draw_circle(r)
