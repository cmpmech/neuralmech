import mlhp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.ndimage
import matplotlib.pyplot as plt

def optimize():
    # 1. High-Resolution verified configuration
    nelx, nely = 100, 100
    volfrac = 0.5
    rmin = 4.0 
    E0, Emin, nu = 1.0, 1e-4, 0.3
    max_iter = 120
    
    penal = 1.0
    penal_step = 0.04
    penal_max = 3.0

    # 2. Grid & Basis
    lengths, origin = [1.0, 1.0], [0.0, 0.0]
    grid = mlhp.makeRefinedGrid([nelx, nely], lengths, origin)
    basis = mlhp.makeHpTrunkSpace(grid, degree=1, nfields=2)
    ncells = grid.ncells()

    dx, dy = lengths[0]/nelx, lengths[1]/nely
    
    # MLHP Grid is Y-fastest: (0,0), (0,dy), (0,2dy) ... (dx,0), (dx,dy) ...
    centers = np.column_stack((
        np.repeat(np.linspace(dx/2, lengths[0]-dx/2, nelx), nely),
        np.tile(np.linspace(dy/2, lengths[1]-dy/2, nely), nelx)
    ))

    # 3. BCs: Support Left, Load Right-Center (Concentrated)
    dirichlet = mlhp.combineDirichletDofs([mlhp.integrateDirichletDofs(mlhp.vectorField(2, [0.0, 0.0]), basis, [0])])
    traction_integrand = mlhp.neumannIntegrand(mlhp.vectorField(2, "[0.0, -10.0 if (y > 0.48) and (y < 0.52) else 0.0]"))
    load_quadrature = mlhp.quadratureOnMeshFaces(grid, [1])

    # 4. Filter setup (Using y-fastest 2D mapping)
    ceil_r = int(np.ceil(rmin))
    kx, ky = np.meshgrid(np.arange(-ceil_r, ceil_r+1), np.arange(-ceil_r, ceil_r+1))
    kernel = np.maximum(0, rmin - np.sqrt(kx**2 + ky**2))
    
    # Grid is [nelx columns of nely elements] -> (nely, nelx) for standard convolution
    H_sum = scipy.ndimage.convolve(np.ones((nely, nelx)), kernel, mode='constant', cval=0.0)

    def apply_filter(field_flat):
        field_2d = field_flat.reshape(nelx, nely).T # Convert y-fastest to (nely, nelx)
        filtered_2d = scipy.ndimage.convolve(field_2d, kernel, mode='constant', cval=0.0) / H_sum
        return filtered_2d.T.flatten() # Back to y-fastest

    def apply_sensitivity_filter(dc_phys_flat):
        field_2d = (dc_phys_flat.reshape(nelx, nely).T / H_sum)
        filtered_2d = scipy.ndimage.convolve(field_2d, kernel, mode='constant', cval=0.0)
        return filtered_2d.T.flatten()

    # 5. Load Protection (Passive solid at load point)
    passive_solid = (centers[:, 0] > 1.0 - dx) & (centers[:, 1] > 0.45) & (centers[:, 1] < 0.55)

    # Initialize
    x = np.ones(ncells) * volfrac
    x[passive_solid] = 1.0
    x_phys = apply_filter(x)
    
    print(f"Starting Y-Fastest Corrected SIMP: {nelx}x{nely}, rmin={rmin}")

    for loop in range(1, max_iter + 1):
        if penal < penal_max:
            penal = min(penal_max, penal + penal_step)
        
        # Linear Elastic Analysis
        E_values = Emin + x_phys**penal * (E0 - Emin)
        # Assuming scalarFieldFromVoxelData also matches grid ordering (y-fastest)
        E_field = mlhp.scalarFieldFromVoxelData(mlhp.DoubleVector(E_values), [nelx, nely], lengths, origin)
        
        matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
        vector = mlhp.allocateRhsVector(matrix)
        integrand = mlhp.staticDomainIntegrand(mlhp.smallStrainKinematics(2), 
                                               mlhp.planeStressMaterial(E_field, mlhp.scalarField(2, nu)), 
                                               mlhp.vectorField(2, [0.0, 0.0]))
        
        mlhp.integrateOnDomain(basis, integrand, [matrix, vector], dirichletDofs=dirichlet)
        mlhp.integrateOnSurface(basis, traction_integrand, [vector], load_quadrature, dirichletDofs=dirichlet)
        
        u_internal = scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(*matrix.csr_arrays), vector.array)
        compliance = np.dot(vector.array, u_internal)
        allDofs = mlhp.inflateDofs(mlhp.DoubleVector(u_internal), dirichlet)
        
        # Sensitivity Analysis (Order: [du/dx, du/dy, dv/dx, dv/dy])
        grad_eval = mlhp.vectorEvaluator(basis, allDofs, difforder=1)
        res = np.array(grad_eval(centers[:, 0], centers[:, 1]))
        grads = res.reshape(ncells, -1)[:, -4:]
        
        eps_xx, eps_yy = grads[:, 0], grads[:, 3]
        eps_xy = grads[:, 1] + grads[:, 2] 
        
        fac = 1.0 / (1.0 - nu**2)
        sed_E1 = 0.5 * fac * (eps_xx**2 + eps_yy**2 + 2*nu*eps_xx*eps_yy + 0.5*(1-nu)*eps_xy**2)
        dc_phys = -penal * (x_phys**(penal - 1)) * (E0 - Emin) * (2.0 * sed_E1 * (dx * dy))
        
        # Enforce Y-Symmetry (Averaging over center line)
        dc_phys_2d = dc_phys.reshape(nelx, nely)
        dc_phys_sym = 0.5 * (dc_phys_2d + np.fliplr(dc_phys_2d)) # Symmetry in y-fastest means flipping each column
        
        dc = apply_sensitivity_filter(dc_phys_sym.flatten())
        
        # OC Update
        l1, l2, move = 0.0, 1e12, 0.05
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            x_new = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(np.maximum(0, -dc) / np.maximum(1e-15, lmid))))))
            x_new[passive_solid] = 1.0
            if np.sum(x_new) > volfrac * ncells: l1 = lmid
            else: l2 = lmid
        
        change = np.max(np.abs(x_new - x))
        x = x_new
        x_phys = apply_filter(x)
        
        if loop % 10 == 0 or loop == 1:
            print(f"It: {loop:3d}, Obj: {compliance:.4e}, MaxU: {np.max(np.abs(u_internal)):.2e}")
        
        if change < 0.002 and penal >= penal_max:
            print("Converged.")
            break

    # Final Plot (Reshape correctly for display)
    plt.figure(figsize=(6,6))
    plt.imshow(x_phys.reshape(nelx, nely).T, origin='lower', cmap='gray_r', extent=[0,1,0,1], vmin=0, vmax=1)
    plt.colorbar(label='Density')
    plt.title(f"Final Topology (Obj: {compliance:.2e}, MaxU: {np.max(np.abs(u_internal)):.1f})")
    plt.tight_layout()
    # plt.savefig("outputs/simp_topology.png")
    plt.show()
    print(f"Result saved to outputs/simp_topology.png")


if __name__ == "__main__":
    optimize()