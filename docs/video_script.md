# 3-minute video storyboard: "GENCO, Certified"

Record: screen capture of the HF Space plus the report figures. Voice-over is in quotes.

| Time | Visual | Voice-over |
|---|---|---|
| 0:00–0:15 | Title card; IEEE 118 grid map from the Space | "GENCO can solve a power flow in one forward pass. Before an operator acts on that answer, they need to know when to trust it." |
| 0:15–0:40 | Fig 1: mismatch halving per layer | "We opened GENCO up. Its twelve layers behave like a learned iterative solver: each one roughly halves its own power mismatch. The last three add very little, so you can stop early." |
| 0:40–1:10 | Fig 2 (dumbbell); then the Space convergence chart, orange dotted vs green | "We also found something hidden. GENCO is asked to *guess* the slack-bus voltage, which is a setpoint the operator already knows. Its answers balance power at its own guess. At the real setpoint the mismatch is 26 to 60 times larger, concentrated right next to the slack bus." |
| 1:10–1:40 | Space: pick an N-5 scenario; green line drops in 1–2 Newton steps; decision card | "The fix is classical. One Newton step from GENCO's answer gives an essentially exact AC solution, and the size of that step tells you how wrong GENCO was. We turn that into a policy: certify after one step, or escalate." |
| 1:40–2:20 | Fig 4 (stress test); highlight zero misses and 40–51% | "We calibrated on single outages only, then tested on double, triple and five-element outages GENCO never saw. The policy found every violation, with no false alarms, at roughly half the factorizations of the industry standard: Newton warm-started from the pre-contingency base case. DC screening misses about half." |
| 2:20–2:45 | Table: GOC 2000 and best-of-two | "And where GENCO isn't the best prior, at two thousand buses with the small released model, a free rule picks the better starting point, so we're never worse than industry practice." |
| 2:45–3:00 | Repo, report, Space links | "Everything is open: the code, a graphkit plugin, datasets, and this live console. GENCO, certified." |
