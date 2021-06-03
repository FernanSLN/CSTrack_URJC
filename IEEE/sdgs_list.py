
file = open("/home/fernan/Documents/Lista de SDGS.txt", "r")
lines = file.readlines()
sdgs_keywords = []

for l in lines:
    sdgs_keywords.append(l.replace("\n", ""))

