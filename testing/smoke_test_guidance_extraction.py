# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
testing/smoke_test_guidance_extraction.py

No-API-key check that the guidance-extraction workflow runs end to end
through both runners and that the evaluator scores the way it claims to.
Only the model is faked: a scripted router answers each prompt the way a
perfect, a trap-prone, or a broken model would.

  perfect      every item and change exactly right (and wrapped in a
               ```json fence, which must still parse): every quality
               dimension 100, critical invariants PASS
  trap-prone   reports the $4.0B Street consensus instead of the company's
               $4.1B, extracts Q3 reported revenue as guidance, misses the
               free-cash-flow withdrawal, and claims Q3 revenue guidance was
               "withdrawn" after Q3 simply ended. Loses points and FAILS the
               critical invariants
  broken       returns prose, not JSON: schema validity 0, critical PASS
               (nothing was claimed, so nothing was hallucinated)

Run from the project root:
    python testing/smoke_test_guidance_extraction.py
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_router import ModelResponse  # noqa: E402
from workflows.guidance_extraction.evaluator import GuidanceExtractionEvaluator  # noqa: E402
from workflows.guidance_extraction.runner_autogen import GuidanceAutoGenRunner  # noqa: E402
from workflows.guidance_extraction.runner_langgraph import GuidanceLangGraphRunner  # noqa: E402
from workflows.guidance_extraction.runner_ensemble import GuidanceEnsembleRunner  # noqa: E402
from workflows.guidance_extraction.task import load_task  # noqa: E402

TASK = "benchmarks/guidance_extraction/northwind_q2_q3_fy2026.yaml"
QUALITY_DIMS = ["schema_validity", "extraction_accuracy", "change_detection", "grounding"]


def perfect_answer(task) -> dict:
    """Built from the answer key, so a perfect score proves the evaluator
    and the key agree with each other."""
    guidance = [{"source": g.source, "metric": g.metric, "period": g.period, "low": g.low, "high": g.high}
                for g in task.expected_guidance]
    changes = [{"metric": c.metric, "period": c.period, "type": c.type, "from": c.from_, "to": c.to}
               for c in task.expected_changes]
    return {"guidance": guidance, "changes": changes}


def trap_prone_answer(task) -> dict:
    a = perfect_answer(task)
    for g in a["guidance"]:
        if (g["source"], g["metric"], g["period"]) == ("q2", "revenue", "FY2026"):
            g["low"] = g["high"] = 4.0                               # Street consensus, not guidance
    a["guidance"].append({"source": "q3", "metric": "revenue", "period": "Q3 FY2026",
                          "low": 1.06, "high": 1.06})                # a reported actual
    a["changes"] = [c for c in a["changes"] if c["metric"] != "free_cash_flow"]   # missed withdrawal
    a["changes"].append({"metric": "revenue", "period": "Q3FY2026", "type": "withdrawn",
                         "from": {"low": 1.0, "high": 1.04}, "to": None})          # period just ended
    return a


class ScriptedRouter:
    """Answers each prompt from a full answer dict, slicing it the way
    each runner step expects (whole answer, one document, or changes)."""

    def __init__(self, answer, raw=None):
        self.answer, self.raw = answer, raw

    def call(self, model_id, prompt):
        if self.raw is not None:
            text = self.raw
        elif "comparing company guidance" in prompt:
            text = json.dumps({"changes": self.answer["changes"]})
        elif "from ONE earnings call" in prompt:
            doc_id = prompt.split('"source": "')[1].split('"')[0]
            text = json.dumps({"guidance": [g for g in self.answer["guidance"] if g["source"] == doc_id]})
        else:
            text = "```json\n" + json.dumps(self.answer, indent=2) + "\n```"
        return ModelResponse(model_id, text, input_tokens=1000, output_tokens=300,
                             latency_ms=1500.0, cost_usd=0.005)


def run_case(name, router, task):
    runners = {
        "autogen": GuidanceAutoGenRunner(router),
        "langgraph": GuidanceLangGraphRunner(router),
        "ensemble": GuidanceEnsembleRunner(router, num_agents=5),
    }
    results = [(fw, "fake-model", r.run("fake-model", task)) for fw, r in runners.items()]
    scored = GuidanceExtractionEvaluator().score_all(results, task)
    print(f"\n== {name} ==")
    for fw, _, _, sr in scored:
        d = sr.dimensions
        print(f"  {fw:<10} composite {sr.composite_score:5.1f}  "
              + "  ".join(f"{k} {d[k]['score']:.0f}" for k in QUALITY_DIMS)
              + f"  critical {'PASS' if d['critical_invariants']['score'] == 100 else 'FAIL'}")
        for k in QUALITY_DIMS + ["critical_invariants"]:
            print(f"      {k}: {d[k]['reason']}")
    return {fw: sr.dimensions for fw, _, _, sr in scored}


def main():
    task = load_task(TASK)
    failures = []

    def expect(cond, msg):
        if not cond:
            failures.append(msg)

    perfect = run_case("perfect", ScriptedRouter(perfect_answer(task)), task)
    for fw, d in perfect.items():
        for k in QUALITY_DIMS:
            expect(d[k]["score"] == 100.0, f"perfect/{fw}: {k} = {d[k]['score']}, expected 100")
        expect(d["critical_invariants"]["score"] == 100.0, f"perfect/{fw}: critical should PASS")

    traps = run_case("trap-prone", ScriptedRouter(trap_prone_answer(task)), task)
    for fw, d in traps.items():
        expect(d["critical_invariants"]["score"] == 0.0, f"trap-prone/{fw}: critical should FAIL")
        reason = d["critical_invariants"]["reason"]
        expect("consensus" in reason and "Q3 reported revenue" in reason,
               f"trap-prone/{fw}: critical reason should name both traps, got: {reason}")
        expect(d["extraction_accuracy"]["score"] < 100, f"trap-prone/{fw}: extraction should lose points")
        expect(d["change_detection"]["score"] < 100, f"trap-prone/{fw}: change detection should lose points")
        expect(d["grounding"]["score"] < 100, f"trap-prone/{fw}: grounding should lose points")

    broken = run_case("broken", ScriptedRouter(None, raw="Revenue guidance went up this quarter."), task)
    for fw, d in broken.items():
        expect(d["schema_validity"]["score"] == 0.0, f"broken/{fw}: schema validity should be 0")
        expect(d["critical_invariants"]["score"] == 100.0, f"broken/{fw}: nothing claimed, critical should PASS")

    # The loader must reject a broken answer key instead of scoring against it.
    import tempfile, yaml
    bad = yaml.safe_load(open(TASK))
    bad["expected_changes"][0]["type"] = "boosted"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(bad, f)
    try:
        load_task(f.name)
        failures.append("loader accepted an unknown change type")
    except ValueError:
        pass
    finally:
        os.unlink(f.name)

    print()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("All guidance-extraction smoke checks passed.")


if __name__ == "__main__":
    main()
