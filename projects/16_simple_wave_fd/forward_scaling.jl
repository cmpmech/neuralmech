const USE_GPU = true #false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using DelimitedFiles
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float32, 2)
else
    @init_parallel_stencil(Threads, Float32, 2)
end
using Plots, Printf, Statistics

@parallel function step_U!(U0::Data.Array, U1::Data.Array, U2::Data.Array, 
                           C_x::Data.Number, C_y::Data.Number)
    @inn(U2) = 2.0*@inn(U1) - @inn(U0) + 
               C_x^2 * (@d2_xi(U1)) +
               C_y^2 * (@d2_yi(U1))    
    return nothing
end

##################################################
@views function acoustic2D(; nx::Int = 255, ny::Int = 255)
    # Physics
    lx, ly    = 1.0f0, 1.0f0  # domain extends
    c         = 1.0f0         # wave speed
    rho       = 1.0f0         # density
    t         = 0.0f0         # physical time
    src_f     = 8.0f0
    # Numerics
    nt        = 1000        # number of timesteps
    # Derived numerics
    dx, dy    = Float32.(lx/(nx-1)), Float32.(ly/(ny-1))       # cell sizes
    dt        = 0.95f0 * Float32.(min(dx,dy)/c)     # CFL condition
    C_x, C_y  = Float32.(c*dt/dx), Float32.(c*dt/dy)           # Courant numbers
    # Array allocations (with ghost cells)
    U0 = @zeros(nx+2, ny+2)
    U1 = @zeros(nx+2, ny+2)
    U2 = @zeros(nx+2, ny+2)

    # Time loop
    for it = 1:nt
        if (it==11) global wtime0 = Base.time() end

        @parallel step_U!(U0, U1, U2, C_x, C_y)

        U0, U1, U2 = U1, U2, U0
        t += dt
    end

    # Performance
    wtime    = Base.time()-wtime0
    A_eff    = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nt-10)                        # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
    @printf("Total steps=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=2))
    @printf("%1.2f billion dofs per s \n", nx * ny * (nt - 10) / 1e9 / wtime)
    return wtime_it
end

wtimes = Float64[]
dofs   = Int[]
nlist = [2^pow for pow in 2:15] # 15
for n in nlist
    wtime = acoustic2D(nx=n, ny=n)
    push!(wtimes, wtime)
    push!(dofs, n*n)
end

data = hcat(dofs, wtimes)
writedlm("output/simple_julia_timings.csv", data, ' ')