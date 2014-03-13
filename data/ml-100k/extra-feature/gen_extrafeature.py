lines = open("u.user", "r").readlines()
lines = [line.split('|')[0:4] for line in lines]

num = 0
occupation_dict = {}
for line in lines:
    occupation = line[3]
    if not occupation_dict.has_key(occupation):
        occupation_dict[occupation] = num;
        num += 1

for line in lines:
    line[3] = str(occupation_dict[line[3]])
    line[2] = "0" if line[2] == 'M' else "1"
    print ','.join(line)


