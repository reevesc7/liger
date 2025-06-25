from liger.output_processing import list_unfinished_runs


OUT_FILE = "unfinished.txt"


unfinished = list_unfinished_runs("Outputs")
with open(OUT_FILE, "w") as f:
    f.writelines("\n".join(unfinished))
print("Wrote to", OUT_FILE)

