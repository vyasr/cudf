# PR Python CUDA coverage

`plan_python_test_coverage.py` consumes wheel and Conda test matrices from one
shared-workflows resolution. Existing job selectors run before the coverage
split; cuDF does not maintain a copy of the shared matrix definitions.

Upstream pandas and Polars use the newest eligible amd64 wheel environment
(latest driver and dependencies), comparing CUDA and Python versions numerically.
The planner derives an explicit container tag from that entry and `VERSION`;
custom jobs retain their RTX Pro runners and shard counts.

When an upstream suite is scheduled, its CUDA major replaces matching amd64
internal runs, except oldest-dependency entries. Pandas substitutes for internal
cuDF, pylibcudf, and cuDF pandas coverage; Polars substitutes for internal
cuDF-Polars coverage. Existing ARM entries remain unchanged. If an exclusion would
remove every internal amd64 entry, the original job matrix is retained instead.
Absent upstream suites do not change their internal matrices.

This tradeoff applies only to PRs. Nightly matrices and the internal earliest-wheel
/latest-Conda Polars-version split remain unchanged. The planning job reports the
selected environment and every retained or removed entry in its workflow summary.

Run the policy tests with:

```sh
python -m unittest discover -s ci/utils -p test_plan_python_test_coverage.py
```
