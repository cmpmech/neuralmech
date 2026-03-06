extern "C"
{

__global__ void addSerial(const float *x, const float *y, float *z, int N){
     for (int i = 0; i < N; i++){
         z[i] = x[i] + y[i];
    }
}

__global__ void addParallel(const float *x, const float *y, float *z, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        z[i] = x[i] + y[i];
    }
}

} // extern "C"
