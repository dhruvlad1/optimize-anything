// bench_redundant.ml - Repeated subexpressions, constant folding, and dead code
i = 0;
total = 0;
while (i < 20000) {
    c1 = 2 * 3 + 4;
    dead1 = i * 42 + 99;
    dead2 = dead1 * 7;
    x = i * 5 + c1;
    y = i * 5 + c1;
    z = (x + y) * (x + y);
    dead3 = z + 100;
    total = total + (x + y);
    i = i + 1;
}
print total;
