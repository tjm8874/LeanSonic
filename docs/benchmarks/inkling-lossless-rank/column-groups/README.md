# Column-conditioned upper ranks

The 607,934,332-byte frame saves 200,622 bytes over the 608,134,954-byte rank frame (0.033% of that frame, 0.0015% of the complete four-bank package). A decoder reading only the new frame reproduced all 463,159,296 BF16 words, including signed zero. The estimated best of 25 splits used one row group and 16 column groups. Group labels, the complete original low-rank payload, rank table, stream lengths and all 16 independently packed upper-rank streams are counted in the file size.

The encoder sorts rows and columns independently by their mean unsigned BF16 magnitude code, bins them into quantiles, and scores candidate group histograms at their entropy plus the estimated metadata cost. This statistic is an ordering proxy for log magnitude. The actual coder uses the existing in-tree tiled rANS implementation. The extra column scatter is exercised by the offline verifier. The small saving does not justify introducing a second format into the production loader at this point. The PR keeps the simpler single-stream frame.

To reproduce, retain fit/embedding.frame, fit/upper and fit/lower from the [original rank encoder](../encode.py) under BASE, together with the verified uncompressed publisher embedding as BASE/head.decoded. Build [pack.rs](../pack.rs) and [unpack.rs](unpack.rs) as release leansonic-core examples. Create an empty output directory OUT and run the following:

```sh
python3 fit.py BASE /absolute/path/to/pack OUT > fit.json
python3 verify.py OUT/context.frame /absolute/path/to/unpack BASE/head.decoded > verify.json
```

The [fit receipt](fit.json) includes all estimated candidates, the actual representation size and SHA-256. The [independent full decode](verify.json) includes the reconstructed weight hash.
