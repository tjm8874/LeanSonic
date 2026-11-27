# BF16 quality on the timed RTX 5070 Ti

The same physical models, GPU and module used for the [timing comparison](README.md) scored 100 chunks and 25,500 WikiText positions per split. LeanSonic used full GPU residency, FP16 KV, compact gate/up and automatic down. The native llama.cpp scorer used -c 512 -b 512 -ub 512, FP16 KV and requested full GPU offload (-ngl 99). The 8192-batch setting used on PRO 5000 exhausted compute scratch on this 16 GB card. Both LeanSonic splits reproduce every statistic from RTX 5060 Ti.

| Split | Engine | Mean KL | Worst KL | Top-1 / 25,500 | PPL |
|---|---|---:|---:|---:|---:|
| test | LeanSonic | 0.2247122 | 6.6547808 | 20,453 | 8.482171 |
| test | llama.cpp IQ1_S | 0.1135430 | 5.3920480 | 21,915 | 7.716134 |
| valid | LeanSonic | 0.2077958 | 6.5760688 | 20,786 | 7.841980 |
| valid | llama.cpp IQ1_S | 0.1072780 | 5.7341140 | 22,181 | 7.117940 |

Use the pinned binaries and models linked in the timing report. For LeanSonic run `leansonic eval-kld --weights MODEL --ref ORACLE --chunks 100`. For llama.cpp run `llama-perplexity -m MODEL -f CORPUS --kl-divergence --kl-divergence-base ORACLE -c 512 -b 512 -ub 512 -ctk f16 -ctv f16 -ngl 99 -t 8 --no-warmup --chunks 100`. [Raw scores and digests](same-card-quality.json)
