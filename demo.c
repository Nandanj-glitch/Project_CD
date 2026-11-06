#include <stdio.h>

// recursion
int fact(int n) {
    if (n <= 1) return 1;
    return n * fact(n - 1);
}

int fib(int n) {
    int a = 0, b = 1;
    for (int i = 0; i < n; i++) {
        int t = a + b;
        a = b;
        b = t;
    }
    return a;
}

int is_prime(int n) {
    if (n < 2) return 0;
    int i = 2;
    while (i * i <= n) {
        if (n % i == 0) return 0;
        i++;
    }
    return 1;
}

int main() {
    printf("5! = %d\n", fact(5));
    printf("fib(10) = %d\n", fib(10));
    printf("2 + 3 * 4 = %d\n", 2 + 3 * 4);   /* folded at compile time */
    printf("primes below 30:");
    for (int i = 0; i < 30; i++) {
        if (!is_prime(i)) continue;
        printf(" %d", i);
    }
    printf("\n");
    int x = 10;
    x += 5; x *= 2; x -= 1;
    printf("x = %d, -7/2 = %d, -7%%2 = %d\n", x, -7 / 2, -7 % 2);
    printf("%d %d\n", (x > 5 && x < 100), (x < 5 || x == 0));
    int n = 5, steps = 0;
    do { n = n * 2; steps++; } while (n < 100);
    printf("do-while: n=%d steps=%d\n", n, steps);
    printf("bits: %d %d %d %d %d\n", 12 & 10, 12 | 3, 6 ^ 3, 1 << 4, ~0);
    if (0) printf("never printed\n");
    return 0;
}
