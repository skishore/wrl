import collections
import multiprocessing
import subprocess

NUM_SEEDS = 100
NUM_THREADS = 10
BUILD_COMMAND = "cargo build --bin wrl-term --release".split()
SIM_COMMAND = "./target/release/wrl-term --sim 1000".split()

def run_seed(seed: int) -> None:
    counts = collections.defaultdict(int)
    command = SIM_COMMAND + ["--seed", str(seed)]
    output = subprocess.check_output(command, stderr=subprocess.PIPE).decode()
    for line in output.splitlines():
        if line.endswith("removed!"):
            species = line.split(":")[1].split("@")[0].strip()
            counts[species] += 1
    return counts


if __name__ == "__main__":
    seeds = list(range(NUM_SEEDS))
    species_counts = collections.defaultdict(lambda: collections.defaultdict(int))

    subprocess.check_output(BUILD_COMMAND, stderr=subprocess.PIPE)
    with multiprocessing.Pool(NUM_THREADS) as p:
        results = p.map(run_seed, seeds)

    for result in results:
        for (species, count) in result.items():
            species_counts[species][count] += 1

    for (species, counts) in sorted(species_counts.items()):
        print(f"{species}:")
        for (count, frequency) in sorted(counts.items()):
            print(f"  {count} deaths: {frequency} times ({int(100 * frequency / NUM_SEEDS)}%)")
