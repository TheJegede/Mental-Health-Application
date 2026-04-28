import json

with open("notebooks/09_chatbot_evaluation.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if '"overall_gate": overall' in src:
        cell["source"] = [
            l.replace('"overall_gate": overall', '"overall_gate": bool(overall)')
            for l in cell["source"]
        ]
        print("Patched overall_gate cell")

with open("notebooks/09_chatbot_evaluation.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Done.")
