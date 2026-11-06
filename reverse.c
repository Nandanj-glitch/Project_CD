#include <stdio.h>

int main() {
    int n = 12345, rev = 0, digitSum = 0;
    while (n > 0) {
        int d = n % 10;
        rev = rev * 10 + d;
        digitSum += d;
        n = n / 10;
    }
    printf("Reversed = %d\n", rev);
    printf("Digit sum = %d\n", digitSum);
    return 0;
}
