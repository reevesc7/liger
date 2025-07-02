FILEPATH = "smallville_846/simulation_test_013_2024-10-24.txt"


with open(FILEPATH, "r", encoding="cp1252") as file:
    lines = file.readlines()
newlines = []
record = True
for index, line in enumerate(lines):
    if "~~~ output" in line:
        record = False
    if record:
        newlines.append(line)
    if not record and line == "\n":
        record = True
with open(FILEPATH, "w", encoding="cp1252") as file:
    lines = file.writelines(newlines)

