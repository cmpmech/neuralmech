import cupy as cp
import time

parallel = True
# parallel = False

if parallel == True:
    add_kernel = cp.RawKernel(open('add.cu').read(), 'addParallel')
else:
    add_kernel = cp.RawKernel(open('add.cu').read(), 'addSerial')

N = int(1e6)
x = cp.random.uniform(-1,1, N, dtype=cp.float32)
y = cp.random.uniform(-1,1, N, dtype=cp.float32)
z = cp.zeros(N, dtype=cp.float32)

if parallel == True:
    threads = (1024,)
    blocks = ((N + threads[0] - 1) // threads[0],)
else:
    threads = (1,)
    blocks = (1,)

cp.cuda.Stream.null.synchronize()
tic = time.time()
add_kernel(blocks, threads,(x, y, z, N))
cp.cuda.Stream.null.synchronize()
toc = time.time()
print(f'elapsed time: {(toc-tic) * 1e3:.2f}ms')