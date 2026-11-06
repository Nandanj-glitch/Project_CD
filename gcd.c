#include <stdio.h>

int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

int main() {
    printf("gcd(48, 18) = %d\n", gcd(48, 18));
    printf("gcd(17, 5) = %d\n", gcd(17, 5));
    return 0;
}
