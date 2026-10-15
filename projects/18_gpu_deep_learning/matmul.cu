extern "C"
{

// C = A B with A, B, C in R^{N x N}, one thread per entry of C
__global__ void matmul(const float* __restrict__ A, const float* __restrict__ B,
                       float* __restrict__ C, const int N){
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < N){
        float sum = 0.f;
        for(int k = 0; k < N; k++){
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
} // extern "C"
