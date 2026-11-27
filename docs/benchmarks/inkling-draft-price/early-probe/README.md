# Try one neural proposal after six priced skips

The original fixed pause missed six of six accepted proposals over nine tokens on the name fixture. This experiment permits a single timed neural round per 24-round pause. It runs after six actual priced skips, at the next eligible opportunity without a preferred copy proposal or legacy pause. It returns early to ordinary drafting only when elapsed neural time per committed token is at least 2% below the scalar estimate. A failed probe consumes the round; subsequent eligible opportunities stay paused until ordinary expiry. No prompt IDs or expected outputs are consulted by the policy.

Runtime revision `8754a2af9bb3312eb495d926a17ddcf68a18bf63`. Follow the [original experiment](../README.md) to prepare the plain and speculative model directories, exact cubin, GPU selection and 12 input files. Use this directory's `lean.rs` as the release example in place of the original, and this directory's `leg.py` and `analyze.py` as the sampler and validator. Keep the same stock versus candidate ABBA/BAAB order and scalar anchor. The sampler enables `LEANSONIC_SPEC_PRICE_PROBE=1`; the stock and scalar arms leave pricing disabled. It records attempted probes and early resumptions after stopping each request timer. Each per-launch .probes file includes the pass, input name, probe count and resumption count. The validator matches these records to the generation results.

The [predeclared experiment](https://github.com/tjm8874/LeanSonic/issues/177#issuecomment-7224219024) uses RTX 5060 Ti and 5070 Ti with a 1% maximum per-input decode regression, the same Japanese reasoning improvement criterion, and exact scalar output checks on both boards.

[Two-board performance, early resumptions and scalar checks](results/).
