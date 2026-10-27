// bench_mixed.ml - If/else branching combined with loop invariants and dead code
i = 0;
sum_even = 0;
sum_odd = 0;
a = 5;
b = 9;
while (i < 20000) {
    invariant = a * 2 + b * 3;
    dead_val = i * 17;
    if (i % 2 == 0) {
        c_even = 10 * 2 + 5;
        expr1 = i + invariant + c_even;
        expr2 = i + invariant + c_even;
        sum_even = sum_even + expr1 + expr2;
    } else {
        c_odd = 3 * 4 + 7;
        expr3 = i * 2 + invariant;
        expr4 = i * 2 + invariant;
        sum_odd = sum_odd + expr3 + expr4;
    }
    i = i + 1;
}
print sum_even;
print sum_odd;
