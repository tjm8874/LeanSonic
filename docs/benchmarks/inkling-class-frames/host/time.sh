set -euo pipefail
cores=$(python3 -c 'import os; print(",".join(map(str,sorted(os.sched_getaffinity(0))[-4:])))')
task() {
 local matrix=$1 leg=$2 mode=$3
 local frame=new; local width=144
 if [[ $mode == control ]];then frame=old;fi
 if [[ $matrix == w2 ]];then width=72;fi
 local stem="$matrix-$leg-$mode"
 /usr/bin/time -f '%M' -o "$stem.rss" taskset -c "$cores" nice -n 15 ./bench "$mode" "$matrix-$frame" "$width" "$matrix-scale" 2>"$stem.us" | sha256sum | cut -d' ' -f1 >"$stem.tmp"
 if [[ $mode != control ]];then cmp "$matrix-control-control.sha" "$stem.tmp";fi
 mv "$stem.tmp" "$stem.sha"
}
for m in w13 w2;do
 task "$m" control control
 for c in 1 2 3 4 5;do
  arms=(old new new old);if ((c%2==0));then arms=(new old old new);fi
  for k in 0 1 2 3;do task "$m" "$c-$k" "${arms[k]}";done
 done
done
touch done
