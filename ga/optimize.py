import math
import random
import re
import shlex
import statistics
import subprocess
import sys

from pathlib import Path

# Where to search for genes, and how to build and run a sim:
BASE_DIR = "./base/src"
BUILD_COMMAND = "cargo build --bin wrl-term --release".split()
SIM_COMMAND = "./target/release/wrl-term --gym 9999".split()

# Number of iterations:
GENERATIONS = 2

# We run many samples because our fitness signal is noisy.
SAMPLES_PER_INDIVIDUAL = 10

# Population counts. Guaranteed winners are passed directly onto the next
# generation without mutation. After that, we fill up to the population size
# by repeatedly sampling 2 "randomized winners" and doing crossover.
POPULATION_SIZE = 10
GUARANTEED_WINNERS = 2
RANDOMIZED_WINNERS = 5

# Mutation strength (sigma for the log-normal exponent). A strength of 0.05
# means a ~5% change is typical (because e^x ~ 1 + x), but this approximation
# breaks down as the value increases.
MUTATION_STRENGTH = 0.25

GENE_REGEX = r"gene!\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)"

VERBOSE = False


class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None
        self.fitness_std = None


def get_source_files():
    """Returns a sorted list of all Rust files to ensure deterministic ordering."""
    files = list(Path(BASE_DIR).rglob("*.rs"))
    files.sort() # Critical for consistency between extract and inject
    return files


def extract_genes(files):
    """Finds all genes across all files in a specific order."""
    genes = []
    for path in files:
        content = path.read_text()
        matches = re.findall(GENE_REGEX, content)
        genes.extend([float(m) for m in matches])
    return genes


def inject_genes(files, genome):
    """Replaces genes in the tree with the given genome, relying on file order."""
    genome_iter = iter(genome)

    # This function consumes from genome_iter for every regex match found.
    # Since the number of genes does not change, do not catch StopIteration.
    def replace_match(match):
        return f"gene!({next(genome_iter):.10f})"

    for path in files:
        old_content = path.read_text()
        new_content = re.sub(GENE_REGEX, replace_match, old_content)
        if new_content != old_content:
            path.write_text(new_content)


def evaluate_individual(files, individual):
    """Injects, builds, and runs trials."""
    # 1. Overwrite genes in the source code.
    inject_genes(files, individual.genome)

    # 2. Rebuild the sim.
    if VERBOSE:
        print(f"Running command: {shlex.join(BUILD_COMMAND)}")
    subprocess.check_output(BUILD_COMMAND, stderr=subprocess.PIPE)

    # 3. Run sims in parallel.
    processes = []
    for _ in range(SAMPLES_PER_INDIVIDUAL):
        if VERBOSE:
            print(f"Running command: {shlex.join(SIM_COMMAND)}")
        processes.append(subprocess.Popen(SIM_COMMAND, stdout=subprocess.PIPE))

    scores = []
    for process in processes:
        (stdout, _) = process.communicate()
        scores.append(float(stdout.strip()))

    if VERBOSE:
        print(scores)

    assert len(scores) > 1
    return statistics.mean(scores), statistics.stdev(scores)


def main():
    files = get_source_files()
    print("Backing up original source...")
    backups = {path: path.read_text() for path in files}

    try:
        initial_genome = extract_genes(files)
        if len(initial_genome) == 0:
            print("No genes found!")
            return

        print(f"Optimizing {len(initial_genome)} genes across {len(files)} files. Initial value: {initial_genome}")

        # Initialize
        population = [Individual(initial_genome)]
        for _ in range(POPULATION_SIZE - 1):
            # gene = gene * exp(N(0, sigma))
            mutated_genome = [
                g * math.exp(random.gauss(0, MUTATION_STRENGTH))
                for g in initial_genome
            ]
            population.append(Individual(mutated_genome))

        for gen in range(GENERATIONS):
            print(f"\n--- Generation {gen} ---")

            for i, ind in enumerate(population):
                if ind.fitness is None:
                    ind.fitness, ind.fitness_std = evaluate_individual(files, ind)
                print(f"Individual: {i:02d} | Mean: {ind.fitness:8.2f} (±{ind.fitness_std:8.2f}) | Genome: {ind.genome}")

            # Sort. For now, higher fitness values are worse (fix that?)
            population.sort(key=lambda x: x.fitness)

            # Reproduction, based on fitness rank
            next_gen = []
            for best_individual in population[:GUARANTEED_WINNERS]:
                next_gen.append(Individual(best_individual.genome))
            while len(next_gen) < POPULATION_SIZE:
                p1, p2 = random.sample(population[:RANDOMIZED_WINNERS], 2)
                # Simple uniform crossover. Genes are not ordered.
                child_genome = [
                    (p1.genome[i] if random.random() > 0.5 else p2.genome[i])
                    for i in range(len(initial_genome))
                ]
                # Mutation: gene = gene * exp(N(0, sigma))
                child_genome = [
                    g * math.exp(random.gauss(0, MUTATION_STRENGTH))
                    for g in child_genome
                ]
                next_gen.append(Individual(child_genome))
            population = next_gen

        print("\nBest individual found:")
        print(population[0].genome)

    finally:
        print("Cleaning up: restoring original source files.")
        for path, content in backups.items():
            path.write_text(content)


if __name__ == "__main__":
    main()
