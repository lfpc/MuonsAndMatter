#include <cuda_runtime.h>
__global__ void test() {}
int main() { test<<<1,1>>>(); return 0; }