#!/usr/bin/env python3
from collections import OrderedDict
import re
import sys
from itertools import product

MAX_ITERATIONS = 50
CONVERGENCE_EPS = 1e-10

def parse_bif(filename):
    text = open(filename, 'r').read()
    var_block_re = re.compile(r'variable\s+"?(\w+)"?\s*\{([^}]*)\}', re.DOTALL | re.MULTILINE)
    variables = OrderedDict()
    for m in var_block_re.finditer(text):
        name = m.group(1)
        block = m.group(2)
        vals = []
        vals_m = re.search(r'type\s+discrete\s*\[\s*\d+\s*\]\s*\{([^}]*)\}', block, re.DOTALL | re.MULTILINE)
        if vals_m:
            vals_text = vals_m.group(1)
            raw = re.split(r'[",\s]+', vals_text)
            vals = [v for v in (x.strip() for x in raw) if v]
        variables[name] = {'values': vals, 'parents': [], 'probabilities': OrderedDict()}

    prob_re = re.compile(
        r'probability\s*\(\s*"?(?P<target>\w+)"?\s*(?:\|\s*(?P<parents>[^)]+))?\)\s*\{(?P<body>.*?)\};',
        re.DOTALL | re.MULTILINE)
    for m in prob_re.finditer(text):
        target = m.group('target')
        parents_text = m.group('parents')
        body = m.group('body').strip()
        parents = []
        if parents_text:
            parents = [p.strip().strip('"') for p in parents_text.split(',') if p.strip()]
        if target not in variables:
            variables[target] = {'values': [], 'parents': parents, 'probabilities': OrderedDict()}
        else:
            variables[target]['parents'] = parents

        if body.startswith('table'):
            nums = re.findall(r'-?\d+\.?\d*', body)
            probs = [float(x) if x != '-1' else -1.0 for x in nums]
            variables[target]['probabilities']['()'] = probs
        else:
            row_re = re.compile(r'\(([^\)]+)\)\s*([^;\n]+)')
            for rm in row_re.finditer(body):
                key_text = rm.group(1)
                key = tuple([k.strip().strip('"') for k in key_text.split(',')])
                nums = re.findall(r'-?\d+\.?\d*', rm.group(2))
                probs = [float(x) if x != '-1' else -1.0 for x in nums]
                variables[target]['probabilities'][key] = probs

    for v, info in variables.items():
        if not info['values']:
            print(f"[WARN] Variable {v} has no domain values parsed — assigning default ['True','False'].")
            variables[v]['values'] = ['True', 'False']

        parents = info['parents']
        if parents:
            parent_domains = [variables[p]['values'] for p in parents]
            for comb in product(*parent_domains):
                if comb not in info['probabilities']:
                    info['probabilities'][comb] = [-1.0] * len(info['values'])
        else:
            if '()' not in info['probabilities']:
                info['probabilities']['()'] = [-1.0] * len(info['values'])

    return variables

def parse_data(bif_data, filename):
    ordered_vars = list(bif_data.keys())
    var_idx = {v: i for i, v in enumerate(ordered_vars)}
    data_comp = []
    data_incomp = []
    with open(filename, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip().strip('"') for p in re.split(r'[,\"]\s*|\s+', line) if p.strip()]
            if len(parts) != len(ordered_vars):
                parts = [p.strip().strip('"') for p in line.split(',') if p.strip()]
            if len(parts) != len(ordered_vars):
                raise ValueError(f"Record length mismatch: expected {len(ordered_vars)} fields but got {len(parts)}. Line: {line}")

            missing = False
            missing_idx = None
            for i, val in enumerate(parts):
                if val == '?':
                    missing = True
                    missing_idx = i
                    break
            if missing:
                data_incomp.append((parts, missing_idx))
            else:
                data_comp.append(parts)
    return data_comp, data_incomp, ordered_vars

def prob_init(network, data_comp, ordered_vars):
    counts = {}
    for var, info in network.items():
        counts[var] = {}
        num_vals = len(info['values'])
        for parent_config in info['probabilities'].keys():
            counts[var][parent_config] = [0] * num_vals

    for rec in data_comp:
        for var in ordered_vars:
            info = network[var]
            num_vals = len(info['values'])
            if num_vals == 0:
                continue
            i = ordered_vars.index(var)
            val = rec[i]
            try:
                ind = info['values'].index(val)
            except ValueError:
                continue
            parent_vals = tuple(rec[ordered_vars.index(p)] for p in info['parents']) if info['parents'] else ()
            key = parent_vals if parent_vals else '()'
            if key not in counts[var]:
                counts[var][key] = [0] * num_vals
            counts[var][key][ind] += 1

    for var, info in network.items():
        num_vals = len(info['values'])
        if num_vals == 0:
            continue
        for parent_key in list(info['probabilities'].keys()):
            cnts = counts[var].get(parent_key, [0] * num_vals)
            total = sum(cnts)
            newprobs = [ (cnt + 1.0) / (total + num_vals) for cnt in cnts ]
            s = sum(newprobs)
            if s <= 0:
                norm = [round(1.0 / num_vals, 10)] * num_vals
            else:
                norm = [round(p / s, 10) for p in newprobs]
            network[var]['probabilities'][parent_key] = norm
    return network

def initialize_exp_counts(network):
    exp_cnts = {}
    for var, info in network.items():
        exp_cnts[var] = {}
        for parent_key in info['probabilities']:
            exp_cnts[var][parent_key] = [0.0] * len(info['values'])
    return exp_cnts

def children(network):
    chil_map = {var: [] for var in network}
    for var, info in network.items():
        for p in info['parents']:
            if p in chil_map:
                chil_map[p].append(var)
    return chil_map

def calc_exp_cnts(network, ordered_vars, data_comp, data_incomp, chil_map):
    exp_cnts = initialize_exp_counts(network)
    var_idx = {v:i for i,v in enumerate(ordered_vars)}

    for rec in data_comp:
        for var in ordered_vars:
            info = network[var]
            if not info['values']:
                continue
            i = var_idx[var]
            val = rec[i]
            try:
                vind = info['values'].index(val)
            except ValueError:
                continue
            parent_vals = tuple(rec[var_idx[p]] for p in info['parents']) if info['parents'] else ()
            key = parent_vals if parent_vals else '()'
            if key not in exp_cnts[var]:
                exp_cnts[var][key] = [0.0] * len(info['values'])
            exp_cnts[var][key][vind] += 1.0

    for rec, missing_idx in data_incomp:
        miss_var = ordered_vars[missing_idx]
        miss_info = network[miss_var]
        m_num = len(miss_info['values'])
        if m_num == 0:
            m_num = 2
            miss_info['values'] = ['True','False']
        scores = [0.0] * m_num

        for m in range(m_num):
            score = 1.0
            parent_vals = []
            for p in miss_info['parents']:
                parent_vals.append(rec[var_idx[p]])
            parent_vals = tuple(parent_vals) if parent_vals else ()
            pkey = parent_vals if parent_vals else '()'
            if pkey in network[miss_var]['probabilities']:
                score *= network[miss_var]['probabilities'][pkey][m]
            else:
                score *= 1.0 / m_num

            for child in chil_map[miss_var]:
                child_info = network[child]
                c_i = var_idx[child]
                cval = rec[c_i]
                try:
                    cind = child_info['values'].index(cval)
                except ValueError:
                    cind = None
                c_parent_vals = []
                for par in child_info['parents']:
                    if par == miss_var:
                        c_parent_vals.append(miss_info['values'][m])
                    else:
                        c_parent_vals.append(rec[var_idx[par]])
                ckey = tuple(c_parent_vals) if c_parent_vals else '()'
                if cind is None:
                    score *= 0.0
                else:
                    prob_list = child_info['probabilities'].get(ckey)
                    if prob_list is None:
                        score *= 1.0 / len(child_info['values'])
                    else:
                        score *= prob_list[cind]
            scores[m] = score

        total = sum(scores)
        if total == 0:
            weights = [1.0 / m_num] * m_num
        else:
            weights = [s / total for s in scores]

        for m in range(m_num):
            w = weights[m]
            parent_vals = []
            for p in miss_info['parents']:
                parent_vals.append(rec[var_idx[p]])
            pkey = tuple(parent_vals) if parent_vals else '()'
            if pkey not in exp_cnts[miss_var]:
                exp_cnts[miss_var][pkey] = [0.0] * len(miss_info['values'])
            exp_cnts[miss_var][pkey][m] += w

        for var in ordered_vars:
            if var == miss_var:
                continue
            info = network[var]
            if not info['values']:
                continue
            v_i = var_idx[var]
            val = rec[v_i]
            try:
                vind = info['values'].index(val)
            except ValueError:
                continue
            for m in range(m_num):
                w = weights[m]
                pvals = []
                for p in info['parents']:
                    if p == miss_var:
                        pvals.append(miss_info['values'][m])
                    else:
                        pvals.append(rec[var_idx[p]])
                pkey = tuple(pvals) if pvals else '()'
                if pkey not in exp_cnts[var]:
                    exp_cnts[var][pkey] = [0.0] * len(info['values'])
                exp_cnts[var][pkey][vind] += w

    return exp_cnts

def calc_probabilities(network, exp_cnts):
    change = 0.0
    for var, info in network.items():
        num_vals = len(info['values'])
        if num_vals == 0:
            continue
        for parent_key, counts in exp_cnts[var].items():
            tot = sum(counts)
            newprobs = [ (counts[i] + 1.0) / (tot + num_vals) for i in range(num_vals) ]  # Laplace
            s = sum(newprobs)
            if s <= 0:
                norm = [round(1.0 / num_vals, 10)] * num_vals
            else:
                norm = [round(p / s, 10) for p in newprobs]
            old = network[var]['probabilities'].get(parent_key)
            if old is None:
                network[var]['probabilities'][parent_key] = norm
                change += 1.0
            else:
                diff = sum(abs(a - b) for a, b in zip(old, norm))
                change += diff
                network[var]['probabilities'][parent_key] = norm
    return network, change

def final_pipeline(network, records_file):
    data_comp, data_incomp, ordered_vars = parse_data(network, records_file)
    print(f"Complete records: {len(data_comp)}, Incomplete: {len(data_incomp)}")
    network = prob_init(network, data_comp, ordered_vars)
    chil_map = children(network)

    prev_change = None
    for it in range(MAX_ITERATIONS):
        print(f"[EM] Iteration {it+1}")
        exp_cnts = calc_exp_cnts(network, ordered_vars, data_comp, data_incomp, chil_map)
        network, change = calc_probabilities(network, exp_cnts)
        print(f"[EM] total L1 change = {change:.12f}")
        if prev_change is not None and abs(prev_change - change) < CONVERGENCE_EPS:
            print("[EM] Converged.")
            break
        prev_change = change
    return network

def write_bif_with_learned_probs(input_bif, network, output_bif):
    with open(input_bif, 'r') as f:
        lines = f.readlines()

    prob_pattern = re.compile(r'probability\s*\(\s*"?(\w+)"?\s*(?:\|\s*([^\)]+))?\s*\)\s*\{', re.MULTILINE)
    output_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = prob_pattern.match(line.strip())
        if m:
            var_name = m.group(1)
            output_lines.append(line)
            i += 1
            block = []
            while i < len(lines) and '};' not in lines[i]:
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
                i += 1

            info = network.get(var_name)
            if info is None:
                output_lines.extend(block)
                continue

            block_text = ''.join(block)
            if 'table' in block_text:
                probs = info['probabilities'].get('()')
                if probs is None:
                    num = len(info['values'])
                    probs = [1.0 / num] * num
                formatted = ' '.join(f"{p:.4f}" for p in probs)
                output_lines.append(f"\ttable {formatted} ;\n")
            else:
                keys_in_block = []
                row_re = re.compile(r'\(([^\)]+)\)')
                for pl in block:
                    rm = row_re.search(pl)
                    if rm:
                        key = tuple(k.strip().strip('"') for k in rm.group(1).split(','))
                        keys_in_block.append(key)
                if not keys_in_block:
                    keys_in_block = [k for k in info['probabilities'].keys() if k != '()']

                for key in keys_in_block:
                    probs = info['probabilities'].get(key)
                    if probs is None:
                        probs = [1.0 / len(info['values'])] * len(info['values'])
                    key_text = ', '.join(key)
                    output_lines.append(f"\t({key_text}) {' '.join(f'{p:.4f}' for p in probs)}\n")
            output_lines.append("};\n")
        else:
            output_lines.append(line)
            i += 1

    with open(output_bif, 'w') as f:
        f.writelines(output_lines)
    print(f"Wrote learned probabilities to {output_bif}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 startup_code.py hailfinder.bif records.dat solved_hailfinder.bif")
        sys.exit(1)
    bif_file = sys.argv[1]
    records_file = sys.argv[2]
    output_bif = sys.argv[3]

    network = parse_bif(bif_file)
    print(f"Parsed {len(network)} variables from {bif_file}")
    learned_network = final_pipeline(network, records_file)
    write_bif_with_learned_probs(bif_file, learned_network, output_bif)