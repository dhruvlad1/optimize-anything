// bench_invariant.ml - Nested loops with loop-invariant expressions
a = 3;
b = 7;
c = 11;
d = 13;
total = 0;
i = 0;
while (i < 200) {
    j = 0;
    while (j < 500) {
        inv = a * b + c * d;
        total = total + inv;
        j = j + 1;
    }
    i = i + 1;
}
print total;
