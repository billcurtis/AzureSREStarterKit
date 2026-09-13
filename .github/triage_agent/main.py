"""Triage Agent — converts SRE-Agent-filed GitHub issues into proposed fixes.

Architecture
------------

1. An issue is opened on the repo (filed by Azure SRE Agent, or by a human
   following the same format).
2. ``.github/workflows/issue-triage.yml`` invokes ``python -m triage_agent``
   with the issue payload.
3. This module:

   a. Loads the issue body.
   b. (Optional) Queries Azure Resource Graph for the affected resource's
      current state, if the issue body includes an ``ARG-QUERY:`` line.
   c. Calls an Azure OpenAI deployment with a strict prompt requesting a
      structured JSON fix proposal.
   d. Writes ``proposal.json`` + a Markdown PR body + an optional patch file
      to the output directory.

4. The workflow then opens a draft pull request from those artifacts.

Authentication
--------------

Uses :class:`azure.identity.DefaultAzureCredential`:

* In GitHub Actions: ``azure/login@v2`` (Workload Identity Federation) sets
  the OIDC env vars; ``DefaultAzureCredential`` picks them up automatically.
* Locally: ``az login`` works — same code path, no config changes.

Required env vars
-----------------

* ``AZURE_OPENAI_ENDPOINT``  — full https://….openai.azure.com URL
* ``TRIAGE_DEPLOYMENT``      — Azure OpenAI **deployment name**
                               (defaults to ``gpt-4o-mini`` if unset; this is
                                NOT the model name, it's the deployment label
                                you chose when deploying the model).

Optional env vars
-----------------

* ``AZURE_SUBSCRIPTION_ID``           — required only if the issue body
                                         contains an ``ARG-QUERY:`` line.
* ``APPLICATIONINSIGHTS_CONNECTION_STRING`` — if set, distributed traces and
                                         exceptions are emitted to App Insights.
* ``ORG_NAME``                         — text to use in the agent's PR body
                                         signature (defaults to "your org").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Optional OpenTelemetry instrumentation. Configure once at import time so
# every Azure SDK + OpenAI call is captured.
if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor()
    except Exception as _otel_err:  # pragma: no cover - best-effort
        print(f"warning: OTel init failed: {_otel_err}", file=sys.stderr)


ORG_NAME = os.environ.get("ORG_NAME", "your org")

SYSTEM_PROMPT = """You are the Triage Agent.

You receive a GitHub Issue filed by Azure SRE Agent (or a human) describing
a problem detected in an Azure resource group. Your job:

1. Read the issue title + body. Identify the affected Azure resource(s) and
   the type of finding (security drift, cost waste, reliability gap,
   storm/scale issue, compliance violation).
2. Read the "current state" JSON the workflow has attached - it's the result
   of an Azure Resource Graph query for that resource. If no state is attached,
   say so in the summary and proceed with what's in the issue body.
3. Propose a concrete fix as:

   a. A short executive summary (2-3 sentences, plain English)
   b. Root cause - what's actually wrong
   c. Proposed fix - the FULL CONTENT of the file the reviewer should commit
      (not a unified diff; the workflow writes "patch" verbatim into "filename")
   d. Risk - what could go wrong if applied as-is
   e. Verification - how a human reviewer can confirm the fix worked

4. Be conservative. NEVER propose anything that:

   - Deletes data
   - Changes production traffic flow without explicit confirmation
   - Modifies auth/RBAC at scope wider than the affected resource

   When in doubt, propose the smallest reversible change.

5. The "filename" field MUST be a relative path under one of:
   infra/, bicep/, terraform/, scripts/, config/
   It MUST NOT contain ".." or start with "/" or ".github/". If you cannot
   pick a safe target path, set "fix" to null.

Output strictly as JSON matching this schema:

{
  "summary": "string",
  "root_cause": "string",
  "fix": {
    "kind": "bicep" | "cli" | "terraform",
    "filename": "string (relative path inside the target repo, see rules above)",
    "patch": "string (the FULL CONTENT of the file — no diff format)"
  },
  "risk": "string",
  "verification": "string",
  "human_review_focus": ["string", ...]
}

Reasoning instructions:

- Think through the classification + remediation carefully internally, but
  output ONLY the JSON object. No prose before or after.
- If the issue doesn't have enough detail to propose a fix, set "fix" to
  null and explain what data you'd need in "summary".
"""


ALLOWED_FIX_DIRS = ("infra/", "bicep/", "terraform/", "scripts/", "config/")


def _normalize_proposal(raw: dict) -> dict:
    """Defensively coerce model output to the expected shape.

    `response_format=json_object` guarantees the response is *valid JSON*, not
    that it matches the schema. The model may return strings where dicts are
    expected, a single string where a list is expected, etc. Normalize so the
    PR-body renderer can't crash on it.
    """
    proposal: dict = {}
    for key in ("summary", "root_cause", "risk", "verification"):
        v = raw.get(key)
        proposal[key] = v if isinstance(v, str) and v.strip() else "(not provided)"

    fix = raw.get("fix")
    if isinstance(fix, dict):
        filename = str(fix.get("filename") or "").strip()
        if not _is_safe_fix_path(filename):
            filename = ""
        proposal["fix"] = {
            "kind": str(fix.get("kind") or "").lower() if fix.get("kind") in ("bicep", "cli", "terraform") else "",
            "filename": filename,
            "patch": str(fix.get("patch") or ""),
        }
    else:
        proposal["fix"] = None

    focus = raw.get("human_review_focus")
    if isinstance(focus, list):
        proposal["human_review_focus"] = [str(f) for f in focus if str(f).strip()]
    elif isinstance(focus, str) and focus.strip():
        proposal["human_review_focus"] = [focus.strip()]
    else:
        proposal["human_review_focus"] = []

    if "raw_output" in raw:
        proposal["raw_output"] = raw["raw_output"]
    return proposal


def _is_safe_fix_path(path: str) -> bool:
    """Reject absolute paths, parent-traversal, and writes to .git / .github."""
    if not path:
        return False
    if path.startswith("/") or ".." in Path(path).parts:
        return False
    lower = path.lower().replace("\\", "/")
    if lower.startswith(".git/") or lower.startswith(".github/") or lower == ".gitignore":
        return False
    return any(lower.startswith(d) for d in ALLOWED_FIX_DIRS)


@dataclass
class IssueContext:
    title: str
    body: str
    number: int
    labels: list[str] = field(default_factory=list)
    azure_state: dict | None = None


def _client() -> tuple[AzureOpenAI, str]:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    # NOTE: deployment != model. Set TRIAGE_DEPLOYMENT to the name you chose
    # when deploying the model in your Azure OpenAI resource.
    deployment = os.environ.get("TRIAGE_DEPLOYMENT") or os.environ.get("TRIAGE_MODEL", "gpt-4o-mini")
    cred = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        cred, "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-10-21",
    )
    return client, deployment


def triage(ctx: IssueContext) -> dict:
    client, deployment = _client()

    user_msg = f"""GITHUB ISSUE
============
Title:  {ctx.title}
Number: #{ctx.number}
Labels: {", ".join(ctx.labels) or "(none)"}

Body:
{ctx.body}

CURRENT AZURE STATE (Resource Graph snapshot)
=============================================
{json.dumps(ctx.azure_state, indent=2, default=str) if ctx.azure_state else "(not attached - proceed with caveats)"}

Produce the JSON triage object now."""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            parsed = {"summary": "Model output was JSON but not an object.", "raw_output": raw}
    except json.JSONDecodeError as e:
        parsed = {
            "summary": f"Triage agent produced non-JSON output: {e}",
            "fix": None,
            "raw_output": raw,
        }
    return _normalize_proposal(parsed)


def _gather_azure_state(resource_query_hint: str) -> dict | None:
    """Run a best-effort Resource Graph query based on hints in the issue body."""
    if not resource_query_hint:
        return None
    try:
        from azure.mgmt.resourcegraph import ResourceGraphClient
        from azure.mgmt.resourcegraph.models import QueryRequest

        sub = os.environ.get("AZURE_SUBSCRIPTION_ID")
        if not sub:
            return {"query": resource_query_hint, "error": "AZURE_SUBSCRIPTION_ID not set"}
        cred = DefaultAzureCredential()
        client = ResourceGraphClient(cred)
        req = QueryRequest(subscriptions=[sub], query=resource_query_hint)
        resp = client.resources(req)
        return {"query": resource_query_hint, "rows": resp.data}
    except Exception as exc:  # noqa: BLE001
        return {"query": resource_query_hint, "error": str(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue-file", required=True, help="Path to JSON file with GitHub issue payload")
    parser.add_argument("--out-dir", default="./out", help="Where to write triage artifacts")
    parser.add_argument("--state-query", default="", help="Optional ARG query to attach")
    args = parser.parse_args(argv)

    issue = json.loads(Path(args.issue_file).read_text())
    ctx = IssueContext(
        title=issue.get("title", ""),
        body=issue.get("body", ""),
        number=int(issue.get("number", 0)),
        labels=[lbl["name"] if isinstance(lbl, dict) else lbl for lbl in issue.get("labels", [])],
        azure_state=_gather_azure_state(args.state_query) if args.state_query else None,
    )

    proposal = triage(ctx)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proposal.json").write_text(json.dumps(proposal, indent=2))

    md = _render_pr_body(ctx, proposal)
    (out_dir / "pr-body.md").write_text(md)

    fix = proposal.get("fix") or {}
    if fix.get("patch") and fix.get("filename") and _is_safe_fix_path(fix["filename"]):
        # Filename has already been validated by _normalize_proposal, but
        # check again here so a hand-edited proposal.json is also safe.
        patch_path = out_dir / Path(fix["filename"]).name
        patch_path.write_text(fix["patch"])

    print(f"✓ Wrote {out_dir / 'proposal.json'} and {out_dir / 'pr-body.md'}")
    return 0


def _render_pr_body(ctx: IssueContext, proposal: dict) -> str:
    fix = proposal.get("fix") or {}
    focus = proposal.get("human_review_focus") or ["(not provided)"]
    focus_md = "\n".join("- " + str(item) for item in focus)
    kind = fix.get("kind", "") if isinstance(fix, dict) else ""
    filename = fix.get("filename", "(none)") if isinstance(fix, dict) else "(none)"
    patch = fix.get("patch", "(no patch generated)") if isinstance(fix, dict) else "(no patch generated)"
    return f"""## Triage Agent Proposal — closes #{ctx.number}

> Generated by the {ORG_NAME} Triage Agent.
> **Human review required before merge.**

### Summary

{proposal.get("summary", "(not provided)")}

### Root cause

{proposal.get("root_cause", "(not provided)")}

### Proposed fix ({kind or "n/a"})

```{kind}
{patch}
```

Target file: `{filename}`

### Risk

{proposal.get("risk", "(not provided)")}

### Verification

{proposal.get("verification", "(not provided)")}

### What the human reviewer should focus on

{focus_md}

---

<sub>This PR was generated automatically. The agent did not deploy anything. Merging this PR is the human-in-the-loop checkpoint.</sub>
"""


if __name__ == "__main__":
    sys.exit(main())
