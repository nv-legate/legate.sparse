import sys

procs = int(sys.argv[1])
# A pre-recorded table of all of the problem sizes that we
# have calculated to be approximately double of each other.
fractions = {
    1: "0.425",
    2: "0.45",
    4: "0.50",
    8: "0.53",
    16: "0.55",
    32: "0.56",
    64: "0.58",
}
print(fractions[procs])
