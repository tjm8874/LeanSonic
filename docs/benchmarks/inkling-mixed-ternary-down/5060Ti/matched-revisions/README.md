# The CUDA module accounts for the observed six-chunk discrepancy

Hold the historical instrumented CLI (5470f09), model, GPU and token histories fixed. Switching from the historical bc526e3 module to a CUDA module built from the CLI source changed all 1,530 full-logit fingerprints and 49 predicted tokens. The largest change in row KL was 1.57904131 at original test chunk 90 position 290 (1.83869048 to 3.41773179). This comparison changes only the module.

The matched-source 5470f09, 6aa971d and 3364f6d pairs then produced identical complete CSV bytes on RTX 5060 Ti. All three share the SHA-256 b070d5610e034fc4ed3fcbbd39d5920cf3727fc7666d7df3a44b73f9d52bf6dd. The conclusion applies to the six-chunk diagnostic on this GPU. Further isolation is needed to attribute the difference to a particular kernel change.

Follow the [fixture and scorer instructions](../row-replay/README.md). Use raw gate/up and down weights (both switches 0) throughout. Build both old_matched components at 5470f09 and both middle_matched components at 6aa971d, using the same instrument.py and compiler flags. The older historical baseline in the first comparison intentionally reproduces the previously measured combination. [Raw row hashes, score summaries and binary identities](results.json)
