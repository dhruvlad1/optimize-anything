// bench_fib.ml - Iterative Fibonacci fib(30)
n = 30;
a = 0;
b = 1;
i = 0;
while (i < n) {
    temp = a + b;
    a = b;
    b = temp;
    i = i + 1;
}
print a;
