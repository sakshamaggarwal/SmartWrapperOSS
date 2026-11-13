# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
workflows/summarization/runner_ensemble.py

Summarizes a document using a parallel Ensemble / Swarm orchestration pattern:
  Phase 1 ("Swarm"):       N specialist personas read the document in parallel
                           and draft concise perspective summaries.
  Phase 2 ("Synthesizer"): 1 synthesizer merges the N specialist drafts into a
                           single comprehensive, non-redundant summary.

Wall-clock `latency_ms` reflects the parallel critical path
(`max(swarm_latencies) + synthesizer_latency`), while token counts and
`cost_usd` sum across all N + 1 LLM calls.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from core.model_router import ModelResponse, ModelRouter
from core.types import TaskResult
from ingestion.document_loader import Chunk
from workflows.base import Workflow

SUMMARY_PERSONAS = [
    "Executive Strategist (key takeaways & high-level thesis)",
    "Technical & Quantitative Analyst (specific numbers, mechanisms & facts)",
    "Risk & Limitations Auditor (caveats, edge cases & trade-offs)",
    "Domain Context Specialist (background context & structural relationships)",
    "Editorial Clarity Specialist (concise synthesis & logical flow)",
]

SWARM_SUMMARIZE_PROMPT = """You are Specialist #{idx}: {persona}.
Summarize the following document clearly and concisely, focusing on the most important facts, numbers, and takeaways from your analytical angle.

DOCUMENT:
{document}"""

SYNTHESIZE_SUMMARIES_PROMPT = """You are a Lead Editorial Synthesizer.
Combine the best insights from the {num_agents} specialist summaries below, verified against the original document, into one cohesive, accurate, and concise final summary. Return only the final summary with no preamble.

ORIGINAL DOCUMENT:
{document}

SPECIALIST DRAFTS:
{drafts}"""


class SummarizationEnsembleRunner(Workflow):
    def __init__(
        self,
        router: ModelRouter,
        num_agents: int = 5,
        max_workers: int = 20,
    ):
        self.router = router
        self.num_agents = max(1, num_agents)
        self.max_workers = max_workers

    def run(self, model_id: str, task_input: List[Chunk]) -> TaskResult:
        document_text = "\n\n".join(c.text for c in task_input)
        personas = [
            SUMMARY_PERSONAS[i % len(SUMMARY_PERSONAS)]
            for i in range(self.num_agents)
        ]

        swarm_responses: List[ModelResponse] = [None] * self.num_agents  # type: ignore
        with ThreadPoolExecutor(max_workers=min(self.max_workers, self.num_agents)) as pool:
            future_to_idx = {
                pool.submit(
                    self.router.call,
                    model_id,
                    SWARM_SUMMARIZE_PROMPT.format(
                        idx=i + 1, persona=persona, document=document_text
                    ),
                ): i
                for i, persona in enumerate(personas)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                swarm_responses[idx] = future.result()

        drafts_block = "\n\n".join(
            f"--- Specialist #{i + 1} ({personas[i]}) ---\n{resp.text}"
            for i, resp in enumerate(swarm_responses)
        )

        final_resp = self.router.call(
            model_id,
            SYNTHESIZE_SUMMARIES_PROMPT.format(
                num_agents=self.num_agents,
                document=document_text,
                drafts=drafts_block,
            ),
        )

        all_steps = swarm_responses + [final_resp]
        swarm_max_latency = max((r.latency_ms for r in swarm_responses), default=0.0)
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
                "swarm_sum_latency_ms": sum(r.latency_ms for r in swarm_responses),
                "synthesizer_latency_ms": final_resp.latency_ms,
            },
        )
