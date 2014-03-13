import random

lines = open("u.data", "r").readlines()
lines = [line.strip() for line in lines]
lines = ['\t'.join(line.split('\t')[0:3]) for line in lines]
random.shuffle(lines)

for line in lines:
    print line
