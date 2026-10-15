extern "C"
{

// direct k x k correlation of an N x M image, one thread per output pixel
__global__ void conv2D(const float* __restrict__ in, const float* __restrict__ w,
                       float* __restrict__ out, const int N, const int M, const int k){
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int r = k / 2;

    if(i < N && j < M){
        float sum = 0.f;
        for(int a = -r; a <= r; a++){
            for(int b = -r; b <= r; b++){
                if(i + a >= 0 && i + a < N && j + b >= 0 && j + b < M){
                    sum += w[(a + r) * k + (b + r)] * in[(i + a) * M + (j + b)];
                }
            }
        }
        out[i * M + j] = sum;
    }
}
} // extern "C"
