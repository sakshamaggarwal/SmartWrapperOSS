# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/guidance_extraction/runner_ensemble.py

Extracts guidance using a parallel Ensemble / Swarm orchestration pattern
(inspired by Ember Swarm):
  Phase 1 ("Swarm"):       N specialist personas analyze the documents in
                           parallel via ThreadPoolExecutor, each producing an
                           independent candidate JSON extraction.
  Phase 2 ("Synthesizer"): 1 consensus merger compares all N candidate
                           extractions against the source documents, resolves
                           disagreements, filters out non-guidance traps
                           (reported actuals, Street consensus), and emits the
                           final JSON.

Contrast with:
  - runner_autogen.py:   2 sequential conversational turns (Analyst -> Reviewer)
  - runner_langgraph.py: per-document extraction nodes -> compare_node -> merge

Because Phase 1 executes concurrently, wall-clock latency (`latency_ms`) is the
critical-path time (`max(swarm_latencies) + synthesizer_latency`), while token
counts and `cost_usd` sum across all N + 1 LLM calls.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from core.model_router import ModelResponse, ModelRouter
from core.types import TaskResult
from workflows.base import Workflow
from workflows.guidance_extraction.task import (
    OUTPUT_FORMAT, RULES, GuidanceTask, documents_block, metrics_block,
)

DEFAULT_SPECIALISTS = [
    "Revenue & Segment Guidance Specialist",
    "Profitability, Operating Margin & EPS Specialist",
    "Cash Flow, CapEx & Free Cash Flow Withdrawal Specialist",
    "Forensic Actuals-vs-Guidance & Street Consensus Trap Auditor",
    "Multi-Quarter Guidance Revision & Change-Type Specialist",
]

EXTENDED_EXPERTS = DEFAULT_SPECIALISTS + [
    "Equity Research Analyst", "SEC Filing Auditor", "FP&A Director",
    "Quantitative Financial Analyst", "Corporate Controller",
    "Earnings Quality Investigator", "Buy-Side Portfolio Manager",
    "Sell-Side Sector Lead", "Credit Risk Analyst", "Valuation Specialist",
]

SWARM_AGENT_PROMPT = """You are Agent #{idx} ({persona}) in a parallel financial-analysis ensemble.
Independently extract forward-looking company guidance and period-over-period guidance changes from the earnings calls below.
Pay strict attention to your specialty ({persona}) while returning the complete JSON extraction.

METRICS
{metrics}

{rules}

DOCUMENTS (earliest first)
{documents}

{output_format}"""

SYNTHESIZER_PROMPT = """You are the Lead Consensus Synthesizer for a {num_agents}-agent financial analysis ensemble.
Below are {num_agents} independent extractions from specialist analysts, alongside the original source documents and rules.

Synthesize the final, authoritative JSON response by:
1. Verifying every guidance item and change against the source documents.
2. Eliminating any reported historical actuals, Street consensus figures, or industry forecasts that a specialist mistakenly included.
3. Ensuring withdrawn guidance and period-end roll-offs follow the rules strictly.

METRICS
{metrics}

{rules}

DOCUMENTS (earliest first)
{documents}

ENSEMBLE CANDIDATE EXTRACTIONS
{candidates}

{output_format}"""


class GuidanceEnsembleRunner(Workflow):
    def __init__(self, router: ModelRouter, num_agents: int = 5, max_workers: int = 20):
        self.router = router
        self.num_agents = max(1, num_agents)
        self.max_workers = max_workers

    def _get_personas(self) -> List[str]:
        personas = []
        for i in range(self.num_agents):
            personas.append(EXTENDED_EXPERTS[i % len(EXTENDED_EXPERTS)])
        return personas

    def run(self, model_id: str, task_input: GuidanceTask) -> TaskResult:
        task = task_input
        common = dict(
            metrics=metrics_block(task),
            rules=RULES,
            documents=documents_block(task),
            output_format=OUTPUT_FORMAT,
        )
        personas = self._get_personas()

        # Phase 1: Parallel Swarm Fan-Out
        swarm_responses: List[ModelResponse] = [None] * self.num_agents  # type: ignore
        with ThreadPoolExecutor(max_workers=min(self.max_workers, self.num_agents)) as pool:
            future_to_idx = {
                pool.submit(
                    self.router.call,
                    model_id,
                    SWARM_AGENT_PROMPT.format(idx=i + 1, persona=persona, **common),
                ): i
                for i, persona in enumerate(personas)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                swarm_responses[idx] = future.result()

        candidates_block = "\n\n".join(
            f"--- Agent #{i + 1} ({personas[i]}) ---\n{resp.text}"
            for i, resp in enumerate(swarm_responses)
        )

        # Phase 2: Consensus Synthesis
        synth_prompt = SYNTHESIZER_PROMPT.format(
            num_agents=self.num_agents,
            candidates=candidates_block,
            **common,
        )
        final_resp = self.router.call(model_id, synth_prompt)

        all_steps = swarm_responses + [final_resp]
        swarm_max_latency = max((r.latency_ms for r in swarm_responses), default=0.0)
        swarm_sum_latency = sum(r.latency_ms for r in swarm_responses)
        critical_path_latency = swarm_max_latency + final_resp.latency_ms

        return TaskResult(
            output=final_resp.text,
            input_tokens=sum(s.input_tokens for s in all_steps),
            output_tokens=sum(s.output_tokens for s in all_steps),
            latency_ms=critical_path_latency,
            cost_usd=sum(s.cost_usd for s in all_steps),
            extra={
                "num_agents": self.num_agents,
                "total_calls": len(all_steps),
                "swarm_max_latency_ms": swarm_max_latency,
                "swarm_sum_latency_ms": swarm_sum_latency,
                "synthesizer_latency_ms": final_resp.latency_ms,
            },
        )
