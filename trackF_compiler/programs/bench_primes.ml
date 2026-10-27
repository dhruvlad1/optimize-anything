// bench_primes.ml - Count primes below 2000 by trial division
count = 0;
n = 2;
while (n < 2000) {
    is_prime = 1;
    d = 2;
    while (d * d <= n) {
        if (n % d == 0) {
            is_prime = 0;
            d = n;
        }
        d = d + 1;
    }
    if (is_prime == 1) {
        count = count + 1;
    }
    n = n + 1;
}
print count;
