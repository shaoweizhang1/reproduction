"""Table 6: parsing time with and without tabling.

Adapted from deepstochlog/tabled_tree_builder.py at release 0.0.1
(github.com/ML-KULeuven/deepstochlog, 2a6982f). Deviations, in both modes: the
legacy nn/2 fact is supplied, repeated inclusion declarations are removed, and
root answers are printed with forall/2 instead of foreach/2. SLD alone drops
the table declaration and the failure-driven branch that only fills tables;
upstream's tabling=False leaves both in place.

The timer covers grammar read/parse/convert, proof search with output and
Python readback. Imports and answer validation are outside it.
"""
import argparse
import collections
import collections.abc
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, stdev

EXPECTED = {1: 10, 3: 95, 5: 1066, 7: 10386, 9: 68298, 11: 416517}
FAILURE = (
    "solve(H) :- include_atom(H),\n"
    "            clause(H,B),\n"
    "            solve(B),\n"
    "            write_ground_clause(H,B), fail."
)
PRINTER = "foreach(solve(Query), write_ground_clause(OriginalQuery, Query))"
PREFLIGHT_LIMIT = 60


def build_parser():
    parser = argparse.ArgumentParser(
        description="Table 6 under the paper's task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("action", choices=["cell", "validate", "batch", "report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--length", type=int, choices=EXPECTED)
    parser.add_argument("--mode", choices=["sld", "slg"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--wall-limit", type=int, default=3600)
    parser.add_argument("--table-space", default="16000000000")
    parser.add_argument("--stack-limit", default="1g")
    return parser


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def definitions(release):
    sys.path.insert(0, str(release))
    if not hasattr(collections, "Sized"):
        collections.Sized = collections.abc.Sized
    from deepstochlog.parser import parse_rules
    from deepstochlog.tabled_tree_builder import PrologSolver, TabledAndOrTreeBuilder

    # Process supervision only; the converter's output is unchanged.
    def run_prolog(self, program):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pl") as source:
            source.write(program)
            source.flush()
            return subprocess.check_output(
                ["swipl", "-q", "-f", source.name, "-g", "main", "-g", "halt"],
                text=True, timeout=60,
            ).splitlines()

    PrologSolver._run_prolog = run_prolog
    return parse_rules, TabledAndOrTreeBuilder


def cell(args):
    parse_rules, Builder = definitions(args.release)
    started = time.perf_counter()
    grammar = (args.release / "examples/mathexpression/mathexpression_notabling.pl").read_text()
    builder = Builder(parse_rules(grammar), tabling=args.mode == "slg")
    solver = builder.solver
    includes = "\n".join(dict.fromkeys(solver.inclusion_definition.splitlines())) + "\n"
    meta = solver.meta_program
    assert meta.count(":- table solve/1.") == 1 and meta.count(FAILURE) == 1
    assert meta.count(PRINTER) == 1
    meta = meta.replace(PRINTER, PRINTER.replace("foreach", "forall"))
    if args.mode == "sld":
        meta = meta.replace(":- table solve/1.", "").replace(FAILURE, "")
    inputs = ",".join("img%d" % i for i in range(1, args.length + 1))
    program = (meta.format(query="expression(_,[%s],[])" % inputs) + includes
               + builder.prolog_program + "\nnn(_,_).\n")
    prepare = time.perf_counter() - started
    source = args.out / "program.pl"
    source.write_text(program)
    assertion = "predicate_property(solve(_),tabled)"
    if args.mode == "sld":
        assertion = "\\+ " + assertion
    with tempfile.TemporaryFile(mode="w+") as output:
        search_start = time.perf_counter()
        subprocess.run(
            ["swipl", "--table_space=" + args.table_space, "--stack-limit=" + args.stack_limit,
             "-q", "-f", str(source), "-g", "assertion((%s))" % assertion,
             "-g", "main", "-g", "halt"],
            stdout=output, check=True,
        )
        search = time.perf_counter() - search_start
        output.seek(0)
        read_start = time.perf_counter()
        lines = [line.replace("\n", "") for line in output]
        readback = time.perf_counter() - read_start
        total = time.perf_counter() - started
    encoded = '["list",' + ",".join('["img%d"]' % i for i in range(1, args.length + 1)) + "]"
    term = r'\["expression",\["([^"_]+)"\],' + re.escape(encoded) + r',\["list"\]\]'
    pattern = re.compile(r"^\[" + term if args.mode == "slg" else r'\["conj",' + term)
    answers = sorted({m[1] for line in lines if (m := pattern.search(line))})
    valid = len(answers) == EXPECTED[args.length]
    save(args.out / "answers.json", answers)
    save(args.out / "result.json", dict(
        status="ok" if valid else "invalid", mode=args.mode, length=args.length,
        seconds=total, prepare_seconds=prepare, search_output_seconds=search,
        readback_seconds=readback, output_lines=len(lines), answer_count=len(answers),
        answer_sha256=digest(json.dumps(answers).encode()),
        program_sha256=digest(program.encode()), table_flag_verified=True,
    ))
    if not valid:
        raise RuntimeError("Incomplete or incorrect answer count")


def supervised(args, length, mode, folder, limit):
    folder.mkdir()
    cmd = [sys.executable, str(Path(__file__).resolve()), "cell",
           "--release", str(args.release), "--out", str(folder),
           "--length", str(length), "--mode", mode,
           "--table-space", args.table_space, "--stack-limit", args.stack_limit]
    begin = time.monotonic()
    with (folder / "stdout.log").open("x") as stdout, (folder / "stderr.log").open("x") as stderr:
        process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    status = "timeout" if code == 124 else "ok" if code == 0 else "error"
    execution = dict(length=length, mode=mode, status=status, exit=code,
                     wall_seconds=time.monotonic() - begin, wall_limit_seconds=limit)
    save(folder / "execution.json", execution)
    print(json.dumps(execution), flush=True)
    return status


def oracle(args, length, folder):
    # Plain Prolog DCG execution, without the DeepStochLog meta-interpreter.
    grammar = (args.release / "examples/mathexpression/mathexpression_notabling.pl").read_text()
    inputs = ",".join("img%d" % i for i in range(1, length + 1))
    program = ("domain(X,L):-member(X,L).\nnn(_,_).\np(_).\n" + grammar
               + "\nmain :- findall(N,expression(N,[%s],[]),Ns), sort(Ns,Sorted), "
                 "forall(member(V,Sorted),(atom_string(V,S),writeln(S))).\n" % inputs)
    path = folder / "oracle.pl"
    path.write_text(program)
    res = subprocess.run(["swipl", "-q", "-f", str(path), "-g", "main", "-g", "halt"],
                         capture_output=True, text=True, timeout=30, check=True)
    (folder / "oracle.stderr").write_text(res.stderr)
    return sorted(set(res.stdout.splitlines()))


def validate(args):
    rows = []
    for length in (1, 3, 5):
        pair = {}
        for mode in ("sld", "slg"):
            folder = args.out / ("%s_l%d" % (mode, length))
            if supervised(args, length, mode, folder, PREFLIGHT_LIMIT) != "ok":
                raise RuntimeError("Short preflight failed: " + str(folder))
            pair[mode] = json.loads((folder / "answers.json").read_text())
        expected = oracle(args, length, args.out / ("sld_l%d" % length))
        assert pair["sld"] == pair["slg"] == expected
        assert len(expected) == EXPECTED[length]
        rows.append(dict(length=length, answers=len(expected), sets_equal=True,
                         direct_dcg_equal=True, runtime_table_flags_verified=True))
    save(args.out / "validation.json", dict(passed=True, checks=rows))
    print("VALIDATION PASSED", flush=True)


def batch(args):
    check = json.loads((args.validation / "validation.json").read_text())
    assert check["passed"] is True
    versions = subprocess.check_output(["swipl", "--version"], text=True).strip()
    save(args.out / "manifest.json", dict(
        protocol="paper-task-aligned, disclosed repairs", swipl=versions,
        python=sys.version, hostname=os.uname().nodename,
        driver_sha256=digest(Path(__file__).read_bytes()),
        repetitions=args.repetitions, lengths=list(EXPECTED),
        wall_limit_seconds=args.wall_limit, table_space=args.table_space,
        stack_limit=args.stack_limit, validation=str(args.validation),
        release=str(args.release),
    ))
    for length in EXPECTED:
        for repeat in range(1, args.repetitions + 1):
            for mode in ("sld", "slg"):
                folder = args.out / ("%s_l%d_r%d" % (mode, length, repeat))
                status = supervised(args, length, mode, folder, args.wall_limit)
                if status == "error":
                    raise RuntimeError("Aborting batch after unexpected failure: " + str(folder))
                if status == "ok":
                    current = json.loads((folder / "answers.json").read_text())
                    for previous in args.out.glob("*_l%d_r*/answers.json" % length):
                        assert current == json.loads(previous.read_text()), "Answer-set mismatch"
    save(args.out / "complete.json", dict(complete=True, cells=2 * len(EXPECTED) * args.repetitions))
    report(args.out)


def report(results):
    print("\nTable 6: seconds, mean +- sample SD over repetitions")
    for length in EXPECTED:
        for mode in ("sld", "slg"):
            scores, statuses = [], []
            for folder in sorted(results.glob("%s_l%d_r*" % (mode, length))):
                status = json.loads((folder / "execution.json").read_text())["status"]
                statuses.append(status)
                if status == "ok":
                    scores.append(json.loads((folder / "result.json").read_text())["seconds"])
            if len(scores) == len(statuses) > 1:
                value = "%.3f +- %.3f" % (mean(scores), stdev(scores))
            else:
                value = " / ".join(statuses)
            print("length=%d %s: %s" % (length, mode.upper(), value), flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.action == "report":
        return report(args.out)
    if args.release is None:
        parser.error("--release is required for " + args.action)
    args.release = args.release.resolve()
    if args.action != "cell":
        args.out.mkdir(parents=True, exist_ok=False)
    {"cell": cell, "validate": validate, "batch": batch}[args.action](args)


if __name__ == "__main__":
    main()
