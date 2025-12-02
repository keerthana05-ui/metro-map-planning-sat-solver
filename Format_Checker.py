import sys
import re
from collections import OrderedDict

def parse_bif(filename):
    text = open(filename, 'r').read()
    var_block_re = re.compile(r'variable\s+"?(\w+)"?\s*\{([^}]*)\}', re.DOTALL | re.MULTILINE)
    variables = OrderedDict()
    for m in var_block_re.finditer(text):
        name = m.group(1)
        block = m.group(2)
        vals_m = re.search(r'type\s+discrete\s*\[\s*\d+\s*\]\s*\{([^}]*)\}', block, re.DOTALL | re.MULTILINE)
        values = []
        if vals_m:
            raw = re.split(r'[",\s]+', vals_m.group(1))
            values = [v for v in (x.strip() for x in raw) if v]
        variables[name] = {'values': values, 'parents': [], 'probabilities': {}}

    prob_re = re.compile(r'probability\s*\(\s*"?(?P<target>\w+)"?\s*(?:\|\s*(?P<parents>[^)]+))?\)\s*\{(?P<body>.*?)\};', re.DOTALL | re.MULTILINE)
    for m in prob_re.finditer(text):
        target = m.group('target')
        parents_text = m.group('parents')
        body = m.group('body').strip()
        parents = []
        if parents_text:
            parents = [p.strip().strip('"') for p in parents_text.split(',') if p.strip()]
        if target not in variables:
            variables[target] = {'values': [], 'parents': parents, 'probabilities': {}}
        else:
            variables[target]['parents'] = parents

        if body.startswith('table'):
            nums = re.findall(r'-?\d+\.?\d*', body)
            probs = [float(x) for x in nums]
            variables[target]['probabilities']['()'] = probs
        else:
            row_re = re.compile(r'\(([^\)]+)\)\s*([^;\n]+)')
            for rm in row_re.finditer(body):
                key_text = rm.group(1)
                key = tuple([k.strip().strip('"') for k in key_text.split(',')])
                nums = re.findall(r'-?\d+\.?\d*', rm.group(2))
                probs = [float(x) for x in nums]
                variables[target]['probabilities'][key] = probs

    return variables

def check_probabilities(bif_data):
    complete, missing = 0, 0
    ok = True
    for v, info in bif_data.items():
        for key in info["probabilities"]:
            row = info["probabilities"][key]
            if any(x == -1 for x in row):
                missing += 1
            else:
                complete += 1
            if abs(sum(row) - 1) >= 0.0001:
                print(f"[WARN] Probabilities for {v} with parents {key} do not sum to 1. They sum to {sum(row)}")
                ok = False
    return ok

def check_format_and_error(base_file, solved_file, gold_file=None):
    base = parse_bif(base_file)
    solved = parse_bif(solved_file)
    gold = parse_bif(gold_file) if gold_file else None

    print("\nChecking probability consistency of solved file...")
    if check_probabilities(solved):
        print("solved network has full CPTs for all variables (sums ok).")
    else:
        print("solved network has problems with CPT sums or missing entries.")

    print("\nChecking format consistency with base file...")
    passed = True
    missing_vars = set(base.keys()) - set(solved.keys())
    extra_vars = set(solved.keys()) - set(base.keys())
    if missing_vars or extra_vars:
        print(f"Variable mismatch: missing {missing_vars}, extra {extra_vars}")
        passed = False

    for v in base:
        if base[v]["parents"] != solved[v]["parents"]:
            passed = False
            print(f"Parent mismatch for {v}")
        if len(base[v]["values"]) != len(solved[v]["values"]):
            passed = False
            print(f"Value mismatch for {v}")

    unlearned = [v for v in solved if any(-1 in row for row in solved[v]["probabilities"].values())]
    if unlearned:
        passed = False
        print(f"Variables with -1 (not learned): {unlearned}")

    if passed:
        print("Format checks passed.")

    if gold:
        print("\nComputing total learning error relative to gold...")
        total_error = 0.0
        n = 0
        for v in gold:
            for key, gold_row in gold[v]["probabilities"].items():
                sol_row = solved[v]["probabilities"].get(key)
                if sol_row is None:
                    sol_row = solved[v]["probabilities"].get(tuple(key))
                if sol_row is not None:
                    n += len(sol_row)
                    total_error += sum(abs(a - b) for a, b in zip(gold_row, sol_row))
        if n > 0:
            print(f"Total learning error: {(total_error / n):.6f}")
        else:
            print("No comparable rows found to compute error.")

if __name__ == "__main__":
    if len(sys.argv) not in (3,4):
        print("Usage: python3 Format_Checker.py base.bif solved.bif [gold.bif]")
        sys.exit(1)
    base_file = sys.argv[1]
    solved_file = sys.argv[2]
    gold_file = sys.argv[3] if len(sys.argv) == 4 else None
    check_format_and_error(base_file, solved_file, gold_file)