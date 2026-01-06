import math
import random
import re
import shlex
import statistics
import subprocess
import sys

from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = "./base/src"
REBUILD_CMD = "cargo build --bin wrl-term --release".split()
RUN_CMD = "./target/release/wrl-term --gym 9999".split()

POP_SIZE = 10
GENERATIONS = 10
SAMPLES_PER_INDIVIDUAL = 100
# Mutation strength (sigma for the log-normal exponent)
# 0.05 means a ~5% change is typical; 0.15 is more aggressive.
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
    """Distributes the genome across files using a single global iterator."""
    genome_iter = iter(genome)
    for path in files:
        content = path.read_text()

        # This function consumes from genome_iter for every regex match found
        def replace_match(match):
            try:
                return f"gene!({next(genome_iter):.10f})"
            except StopIteration:
                return match.group(0)

        new_content = re.sub(GENE_REGEX, replace_match, content)
        if new_content != content:
            path.write_text(new_content)

def evaluate_individual(files, individual):
    """Injects, builds, and runs trials."""
    # 1. Update Source
    inject_genes(files, individual.genome)

    # 2. Build
    if VERBOSE:
        print(f"Running command: {shlex.join(REBUILD_CMD)}")
    subprocess.check_output(REBUILD_CMD, stderr=subprocess.PIPE)

    # 3. Run Samples in Parallel
    processes = []
    for _ in range(SAMPLES_PER_INDIVIDUAL):
        if VERBOSE:
            print(f"Running command: {shlex.join(RUN_CMD)}")
        processes.append(subprocess.Popen(RUN_CMD, stdout=subprocess.PIPE))

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

        print(f"Optimizing {len(initial_genome)} parameters across {len(files)} files. Initial value: {initial_genome}")

        # Population Init
        population = [Individual(initial_genome)]
        for _ in range(POP_SIZE - 1):
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

            # Sort (Minimization)
            population.sort(key=lambda x: x.fitness)

            # Selection/Evolution
            next_gen = []
            for best_individual in population[:2]:
                next_gen.append(Individual(best_individual.genome))

            while len(next_gen) < POP_SIZE:
                p1, p2 = random.sample(population[:5], 2)
                # Uniform Crossover
                child_genome = [
                    (p1.genome[i] if random.random() > 0.5 else p2.genome[i])
                    for i in range(len(initial_genome))
                ]
                # gene = gene * exp(N(0, sigma))
                child_genome = [
                    g * math.exp(random.gauss(0, MUTATION_STRENGTH))
                    for g in child_genome
                ]
                next_gen.append(Individual(child_genome))

            population = next_gen

        print("\nBest individual found:")
        print(population[0].genome)

    finally:
        print("Cleaning up: Restoring original source files.")
        for path, content in backups.items():
            path.write_text(content)

if __name__ == "__main__":
    main()
