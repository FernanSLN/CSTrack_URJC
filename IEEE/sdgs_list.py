
file = open("/home/fernan/Documents/sdg_keys.txt", "r")
lines = file.readlines()
sdgs_keywords = []

for l in lines:
    sdgs_keywords.append(l.replace("\n", ""))

