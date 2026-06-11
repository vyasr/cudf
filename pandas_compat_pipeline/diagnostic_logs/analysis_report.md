# Pipeline Failure Analysis Report
Generated: 2026-06-10
Groups analyzed: 10

## Executive Summary

The 10-group instrumented mini-pilot did not fail because the model produced bad cudf patches.
It failed earlier: no LLM attempt in this sample produced a `write_file` tool call.
Eight groups entered the LLM tool loop, emitted prose diagnostics, and stopped with `tool_calls_parsed: []` and `wrote_patch: false`.
Two groups bypassed LLM attempts as stale-xfail candidates (`baseline_all_passed: true`, `attempts_detail: []`), but the host verification still rejected them.

The dominant operational signal is the verification wrapper reporting `Flakiness rerun 1/3 failed` for every group.
That prefix is misleading in this pilot: the underlying failures are deterministic verification-mode failures, not evidence that the pandas tests are flaky.
The actual underlying longreprs fall into four recurring buckets:

- `XPASS(strict)` after stale xfail handling for interval constructor tests.
- `ImportError while loading conftest` under `CUDF_PANDAS_FAIL_ON_FALLBACK=1`, caused by fallback on `import_optional_dependency`.
- Collection skip/no-collector behavior for modules with missing optional dependencies or skipped plotting/excel setup.
- Node ID drift or collection mismatch, shown as `not found`, `found no collectors`, or `deselected`.

The pipeline currently treats all of these as equivalent failed reruns.
That collapses distinct classes of work into one ambiguous rejection string and pushes the fixer toward source-code fixes even when the correct lever is xfail-marker handling, dependency/configuration handling, node-ID refresh, or verification policy.

## Category Distribution

| Category | Count | % |
|---|---:|---:|
| No patch produced | 8 | 80% |
| Patch + fallback failure | 0 | 0% |
| Patch + still fails | 0 | 0% |
| Tool parse failure | 0 | 0% |
| Stale xfail only | 2 | 20% |
| Human-flagged | 0 | 0% |

Classification notes:

- The eight `No patch produced` groups each have one `attempts_detail` entry, `wrote_patch: false`, and `tool_calls_parsed: []`.
- Their `raw_llm_response` fields are non-empty, but the text is ordinary prose such as "Let me investigate" or "Let me remove" rather than a malformed JSON/tool payload.
- Therefore these are classified as no patch/no tool execution, not JSON parse failures.
- The two `Stale xfail only` groups have `baseline_all_passed: true`, `xfail_removed: true`, no LLM attempts, and verification failures after the stale entry path.
- No diagnostic shows a model-written patch, so the patch-related categories are empty.

## Per-Group Summaries

| Test | Category | Key Finding |
|---|---|---|
| `tests/indexes/interval/test_constructors.py::TestFromArrays::test_constructor_pass_closed` | No patch produced | Baseline had `xpassed` parametrizations and the LLM diagnosed stale xfail, but it only wrote prose (`Let me search...`) and made no `write_file` call. Verification then showed `XPASS(strict)` plus fallback-on-conftest failures. |
| `tests/indexes/interval/test_constructors.py::TestFromBreaks::test_constructor_pass_closed` | No patch produced | Same stale-xfail shape as `TestFromArrays`: all five params look like passing xfails, but no patch was produced. Verification failed first on `XPASS(strict)` and later on `import_optional_dependency` fallback. |
| `tests/plotting/test_datetimelike.py::TestTSPlot::test_pickle_fig` | No patch produced | LLM diagnosed a cudf.pandas pickling/proxy issue from the reason string, but verification was actually dominated by collection skip/no-collector output and the LLM never called tools. |
| `tests/io/test_sql.py::test_sqlalchemy_integer_mapping` | No patch produced | LLM correctly noticed `collected 1 item / 1 deselected / 0 selected`, likely due to marker filtering such as `single_cpu`, but stopped after prose investigation. All 90 verification result records failed as deselected reruns. |
| `tests/io/test_sql.py::test_read_sql_dtype_backend` | No patch produced | Same SQL collection/filtering pattern as integer mapping: requested node IDs were deselected, not assertion-failing. LLM identified the anomaly but did not emit a tool call. |
| `tests/arrays/test_datetimelike.py::TestDatetimeArray::test_searchsorted_castable_strings` | No patch produced | Verification says `not found` / `no match in any of [<Class TestDatetimeArray>]`, implying stale node IDs or vendored pandas test drift. LLM diagnosed non-existent test but did not patch or flag. |
| `tests/io/excel/test_odswriter.py::test_cell_value_type` | No patch produced | LLM identified `pytest.importorskip("odf")` / missing `odfpy` as an environment gap, but dependency changes are prohibited to the fixer and it did not produce a permitted patch or human flag. |
| `tests/plotting/test_datetimelike.py::TestTSPlot::test_line_plot_inferred_freq` | No patch produced | Verification says `found no collectors` for the requested test, suggesting node-ID drift or collection-time skip. LLM planned to search source but did not call tools. |
| `tests/internals/test_internals.py::TestCanHoldElement::test_interval_can_hold_element` | Stale xfail only | Host removed stale xfail and baseline passed/skipped all target params, but verification still failed in the fail-on-fallback mode at conftest import. No LLM attempt was made. |
| `tests/internals/test_internals.py::TestCanHoldElement::test_period_can_hold_element` | Stale xfail only | Same stale-xfail-only pattern as interval can-hold: first verification records passed/skipped, then fail-on-fallback conftest import rejected the group. |

## Dominant Failure Pattern

The dominant failure pattern is not a cudf API implementation defect.
It is a pipeline control-plane failure around verification interpretation and LLM tool-loop completion.

Every diagnostic includes the prefix `Flakiness rerun 1/3 failed` in rejection output.
The config confirms `flakiness_reruns: 3`, so the verifier reruns target tests three times and fails the group if an early rerun fails.
However, the longreprs show that rerun 1 is not exposing random test instability.
It is surfacing deterministic conditions:

1. XPASS under strict xfail handling.
2. No collectors due to collection-time skip or stale node IDs.
3. Deselection by test markers.
4. Conftest import failure when fallback is forbidden.

This matters because the phrase "flaky failure" sends the wrong signal to both humans and the LLM.
The verifier is acting as a multi-mode gate, but the rejection reason only exposes the first failed rerun label instead of classifying the actual failure cause.
The LLM then starts reasoning from ambiguous baseline snippets and often decides it should investigate further.
Because the tool contract requires explicit parsed tool calls, those prose-only "Let me investigate" responses become terminal no-patch attempts.

## Root Cause Analysis

### 1. No model-written patches occurred

Eight groups have exactly one captured LLM attempt.
In every one of those attempts:

- `wrote_patch` is `false`.
- `tool_calls_parsed` is `[]`.
- `tool_results_full` is `[]`.
- `raw_llm_response` is non-empty prose.

The model was not failing after applying an incorrect cudf fix.
It was failing to enter the required tool protocol.
Examples:

- Interval constructors: "Let me remove the stale xfail entries..." but no `write_file` call.
- SQL groups: "Let me examine the test infrastructure..." but no `read_file` or other parsed tool call.
- Datetimelike arrays: "Let me investigate further." but no action.
- ODS writer: "Let me verify this is a dependency/environment gap issue..." but no permitted outcome.

This points to a prompt/protocol mismatch more than a debugging-skill problem.
The embedded skill tells a human-like agent to investigate step by step, while the pipeline harness needs a strict machine-readable tool call or an explicit terminal classification.
The current `MAX_INVESTIGATION_CYCLES_WITHOUT_PATCH = 6` constant is not the limiting factor in the captured diagnostics; each fresh `FixerAgent` call records only the last attempt, and the observed attempts never made it past the first prose response.

### 2. Verification conflates target mode, fallback mode, and flakiness mode

The stale xfail groups are especially revealing.
For `test_interval_can_hold_element`, the first 24 verification result records are `passed` with longreprs showing the tests were collected and skipped cleanly with exit value 0.
The next 24 records fail before the test body while loading pandas `conftest.py`.
For `test_period_can_hold_element`, the same 8-pass / 8-fail split appears.

The failing longrepr is consistent across groups:

```text
ImportError while loading conftest ...
conftest.py:95: in <module>
    pytz = import_optional_dependency("pytz", errors="ignore")
...
NotImplementedFallbackError: Falling back to the slow path.
The function called was import_optional_dependency.
```

That is not a failure of the target stale-xfail removal.
It is a global incompatibility between `CUDF_PANDAS_FAIL_ON_FALLBACK=1` and pandas' test harness import path.
When fail-on-fallback is enabled, importing pandas' own `conftest.py` calls `import_optional_dependency`, which goes through cudf.pandas and falls back.
The fallback exception occurs before the target test semantics can be evaluated.

### 3. XPASS(strict) appears where stale xfail handling is incomplete

The interval constructor diagnostics are critical:

```text
[XPASS(strict)] TODO: Add a reason for failure
```

Both `TestFromArrays::test_constructor_pass_closed` and `TestFromBreaks::test_constructor_pass_closed` show baseline `xpassed` outcomes.
The LLM correctly diagnosed stale xfails.
But `xfail_removed` is still `false`, and `wrote_patch` is false.
Verification then runs with strict xfail semantics and converts the stale xfail into a failure.

This means stale-xfail detection by the model is not enough.
The pipeline needs a reliable host-side stale-xfail path when baseline results are `xpassed`, or the LLM must be forced to issue a concrete `write_file` call before returning.
Otherwise XPASS(strict) will keep masquerading as a flaky verification failure.

### 4. Collection and node-ID problems are first-class categories, not cudf bugs

Several groups never reach the cudf behavior under test.

SQL examples:

```text
collected 1 item / 1 deselected / 0 selected
```

This indicates marker filtering, likely from the runner's default marker expression (`not slow and not single_cpu and not db and not network`) or test-level marks.
Treating deselection as a target-test failure invites pointless cudf patches.

Plotting and ODS examples:

```text
collected 0 items / 1 skipped
ERROR: found no collectors for ...
```

The ODS writer diagnosis points to `pytest.importorskip("odf")`, making this an environment/dependency gap.
The plotting groups may be similar collection-time skips or stale node IDs; the diagnostic text is insufficiently normalized to distinguish them cleanly.

Datetime array example:

```text
ERROR: not found: ...::TestDatetimeArray::test_searchsorted_castable_strings
(no match in any of [<Class TestDatetimeArray>])
```

This is strongly suggestive of stale test inventory or pandas version/test-tree drift.
It should not enter the cudf fixer loop as if it were a pandas-compat behavior mismatch.

### 5. Context truncation is a secondary but real problem

Several `messages_before` entries are truncated at 2000 characters in the diagnostic render.
The raw user prompts likely contained baseline snippets, test source, and xfail metadata, but the diagnostic output does not preserve enough of them to audit exactly what the LLM saw.
Even so, the observed `raw_llm_response` texts show the model had enough information to name the class of problem.
The bigger issue is that after naming the class, it did not take a protocol-valid terminal action.

## Per-Group Analysis Details

### Interval constructor groups

The model tried to fix stale xfail entries in `python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py`.
That is the right file for stale xfail cleanup.
The diagnosis was directionally correct because baseline results were `xpassed` and verification showed `XPASS(strict)`.
The failure was procedural: no `write_file` call was parsed and `xfail_removed` stayed false.
These groups need either automatic stale-xfail removal when baseline is all XPASS, or a tool-loop constraint that refuses final prose without a patch/human flag.

### SQL groups

The model did not try a cudf code patch.
It identified deselection, which is the important finding.
The modified file list contains the plugin path because the host had already removed an xfail entry, but no model patch occurred.
The right next step is not a cudf implementation fix; it is test-selection policy: determine whether single-CPU/db/network-marked tests should be skipped before fixer scheduling, xfailed with a clear reason, or run under a different marker expression.

### Plotting groups

Both plotting groups hit `found no collectors` in verification, and both later hit the global fail-on-fallback conftest error.
The pickle group has an original reason about `pandas.Period` pickling, but the actual verification never reaches that semantic failure.
The diagnosis may be partly speculative because collection failed first.
The right lever is to normalize collection failures and refresh/validate node IDs before asking the LLM to fix cudf.pandas proxy behavior.

### Datetimelike array group

The test method cannot be found in the vendored pandas tests.
This is likely stale inventory, pandas version drift, or grouping metadata that no longer matches the checkout.
The model diagnosed that accurately but did not flag it as a host/pipeline issue.
This class should be filtered before fixer invocation or converted to a human-actionable inventory refresh report.

### ODS writer group

The LLM identified a concrete dependency gap: `odf` / `odfpy` missing, causing `pytest.importorskip("odf")` to skip collection.
The embedded skill's dependency fix path exists for humans, but the pipeline prompt prohibits adding dependencies.
That conflict leaves the LLM with no permitted patch path.
The pipeline should either allow a dependency-change workflow for this category or classify it as environment-gap/human review before the fixer loop.

### Internals stale-xfail groups

These are the clearest examples that the target change succeeded locally but verification policy still failed the group.
`baseline_all_passed: true`, `xfail_removed: true`, and `attempts_detail: []` mean the stale-xfail path did not need LLM debugging.
The first verification mode passed/skipped the targets.
The failure came from the global `CUDF_PANDAS_FAIL_ON_FALLBACK=1` import path, not from the target tests.

## Recommended Levers (Ranked by Expected Impact)

1. **Classify verification longreprs before labeling them flaky** — Rationale: All 10 groups were rejected behind the same `Flakiness rerun 1/3 failed` prefix, but the underlying causes are XPASS, deselection, no collectors, stale node IDs, and fail-on-fallback conftest import errors. A small classifier would immediately route work to the right lever.

2. **Separate fail-on-fallback verification from pandas test harness import/setup** — Rationale: `CUDF_PANDAS_FAIL_ON_FALLBACK=1` currently fails at `import_optional_dependency` in pandas `conftest.py`, before target tests run. Either whitelist/ignore setup-time fallback for test harness imports or run fail-on-fallback only around the target test body after collection succeeds.

3. **Add a host-side stale-XPASS cleanup path** — Rationale: When baseline outcomes are all `xpassed`, the host can remove xfail entries deterministically or require a structured stale-xfail tool call. Relying on prose LLM output leaves `XPASS(strict)` failures unresolved.

4. **Make the fixer protocol strict: no final prose without a parsed action** — Rationale: Eight attempts stopped after "Let me investigate" responses. The harness should reprompt with a protocol error, count it as `tool_parse_or_missing_tool`, or require one of: `write_file`, `flag_for_human`, or `no_patch_reason` with a recognized category.

5. **Pre-filter collection failures before LLM scheduling** — Rationale: `found no collectors`, `not found`, `no tests ran`, and `deselected` are pipeline/test-inventory states, not cudf behavior. Fixing them upstream saves LLM calls and avoids fake cudf tasks.

6. **Refresh/validate node IDs against the actual vendored pandas checkout** — Rationale: The datetimelike array group points to a non-existent method in the checked-out tests. Node-ID inventory should be generated from current collection, not stale plugin entries alone.

7. **Resolve dependency-gap policy for pandas optional dependencies** — Rationale: ODS writer requires `odfpy`; the skill permits dependency updates in a human workflow, but the fixer prompt prohibits dependency changes. The pipeline needs an explicit environment-gap output path.

8. **Preserve full diagnostic context for audit** — Rationale: The diagnostic JSON contains valuable summaries, but rendered message content is truncated. Keep full prompts/responses in sidecar files or store hashes plus full artifacts so follow-on debugging can verify what the model actually saw.

## Next Steps

1. Add a verification-result classifier that maps longreprs into `xpass_strict`, `fail_on_fallback_setup`, `deselected`, `no_collectors`, `node_not_found`, `dependency_skip`, and `assertion_failure`.

2. Change final rejection reasons to report the classified root cause before the flakiness prefix; for example: `verification_class=fail_on_fallback_setup; function=import_optional_dependency`.

3. Treat all-`xpassed` baseline groups as stale-xfail candidates automatically; do not send them through the general cudf fixer unless xfail removal fails to parse or verify in standard mode.

4. For stale-xfail verification, consider standard-mode success sufficient when fail-on-fallback fails during pandas `conftest.py` import rather than during target test execution.

5. Before creating a fixer task, run/consult collection metadata for every node ID and drop or separately report groups that are deselected, skipped at module import, or absent from the current checkout.

6. Add a model-output guard: if `raw_llm_response` is non-empty and `tool_calls_parsed` is empty, immediately reprompt with a concise protocol error instead of ending the attempt.

7. Add a terminal `flag_for_human` or `environment_gap` action available to the fixer when the skill-required fix is disallowed by the pipeline prompt.

8. For SQL tests, decide whether `single_cpu`/db-marked tests belong in this pipeline. If not, skip them before grouping; if yes, run them with a marker expression that selects them intentionally.

9. For ODS/excel tests, compare pandas CI optional dependency sets against the conda environment and either install `odfpy` in `test_cudf_pandas_pandas_tests` or skip these groups before fixer scheduling.

10. For plotting datetimelike tests, validate whether the module-level skip is dependency-driven, backend-driven, or stale-node-driven before pursuing proxy/pickle fixes.

11. For the datetimelike array `searchsorted_castable_strings` group, regenerate the pandas test inventory from the checked-out pandas 3.0.3 tests and remove stale node IDs from the pipeline input/plugin metadata.

12. Re-run a small pilot only after the classifier and strict tool-protocol guard are in place; otherwise the next pilot will likely reproduce the same no-patch/prose-only failure mode.

## Concrete Follow-On Acceptance Criteria

- Stale xfail groups that pass standard verification should not be rejected solely because pandas `conftest.py` falls back during fail-on-fallback setup.
- Groups whose node IDs are absent or deselected should be reported as inventory/selection issues before LLM invocation.
- LLM attempts with prose but no parsed tool calls should be counted separately from model diagnosis failures.
- Reports should distinguish a real cudf patch failure from a missing optional dependency or collection skip.
- The next diagnostic pilot should include at least one successful `write_file` call or an explicit structured no-patch/human-review reason for each attempted group.
