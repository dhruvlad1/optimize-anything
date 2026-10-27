// bench_sum.ml - Compute sum of 1..100000
i = 1;
s = 0;
while (i <= 100000) {
    s = s + i;
    i = i + 1;
}
print s;
