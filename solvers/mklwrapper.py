# forked from mlhp's ctypes bindings to MKL pardiso/ilu/spmv (MIT, Philipp Kopp;
# upstream now lives in solvers/mlhp_source/mlhp/src/python/mlhp/mkl.py). local
# additions on top of the upstream version: scipy csr input (real and complex,
# 64-bit indices), symmetricHalf override, refactorize() on a fixed sparsity, and
# multiple right-hand sides

import ctypes
import glob
import os
import re
import site
import sys

import mlhp
import numpy
import scipy.sparse


def loadMkl(*, mklRtPath=None, oneApiRoot=None, verbose=False):
    path = mklRtPath

    def printVerbose(message):
        if verbose:
            print("loadMkl: " + message)

    # helper functions for oneAPI installation
    def findLibrary(folder):
        files = glob.glob(os.path.join(folder, "*mkl_rt.*.2"))
        printVerbose(f"Number of files matching: {len(files)}")
        for file in files:
            printVerbose(f"Matching file: {file}")
        return files[0] if len(files) else None

    def findOneApi(root):
        printVerbose(f"Looking in oneAPI root at {root}")
        return findLibrary(os.path.join(root, "mkl", "latest", "lib", "intel64"))

    # helper function for python mkl package
    def findRecord(*args):
        directoryPath = os.path.join(*args, "mkl-*.dist-info")
        printVerbose(f"Searching at {directoryPath}")
        record = glob.glob(os.path.join(directoryPath, "RECORD"))
        if len(record) != 0:
            with open(record[0]) as file:
                relative = re.findall("(.*mkl_rt.*?),", file.read())
                filepath = (
                    glob.glob(os.path.join(directoryPath, "..", relative[0]))
                    if len(relative)
                    else []
                )
                if len(filepath):
                    printVerbose(f"Found path: {filepath[0]}")
                    return filepath[0]

    # try to locate mkl_rt
    if path is None and oneApiRoot is not None:
        printVerbose("Checking oneApiRoot function argument")
        path = findOneApi(oneApiRoot)

    if path is None and "ONEAPI_ROOT" in os.environ:
        path = findOneApi(os.environ["ONEAPI_ROOT"])

    if path is None:
        path = findRecord(sys.prefix, "[lL]ib", "site-packages")

    if path is None:
        path = findRecord(sys.prefix, "[lL]ib", "[pP]ython*", "site-packages")

    if path is None:
        path = findRecord(site.USER_SITE)

    # Load mkl library
    if path is None:
        raise ValueError("Unable to locate mkl_rt library. Did you pip install mkl?")
    else:
        return ctypes.cdll.LoadLibrary(path)
        printVerbose(f"Loading {path}")


class pardisoFactorize:
    def __init__(
        self,
        matrix,
        *,
        messageLevel=0,
        symmetric=None,
        symmetricHalf=None,
        mklLibrary=None,
    ):
        self.mkl = loadMkl() if mklLibrary is None else mklLibrary
        self.pt = (ctypes.c_int64 * 64)()
        self.symmetric = symmetric
        self.messageLevel = ctypes.c_int64(messageLevel)
        self._setMatrix(matrix, symmetricHalf)
        self.iparm = (ctypes.c_int64 * 64)()
        self.iparm[0] = 1  # do not use default values
        self.iparm[1] = 3  # use parallel nested dissection with metis
        self.iparm[9] = 8  # pivoting perturbation: 10^-8
        self.iparm[23] = (
            1  # parallel factorization control: 1->improved two-level factorization algorithm
        )
        self.iparm[26] = 0  # do not perform matrix consistency check
        self.iparm[34] = 1  # zero-based indexing
        self._callPardiso(12)

    def _setMatrix(self, matrix, symmetricHalf=None):
        if hasattr(matrix, "data_address"):  # mlhp sparse matrix (real double)
            self.matrix = matrix
            self.complex = False
            self.shape = matrix.shape
            self.symmetricHalf = (
                matrix.symmetricHalf if symmetricHalf is None else symmetricHalf
            )
            self.addresses = (
                matrix.data_address,
                matrix.indptr_address,
                matrix.indices_address,
            )
        else:  # scipy sparse matrix (real or complex), converted to csr with 64-bit indexing
            csr = scipy.sparse.csr_matrix(matrix)
            self.complex = numpy.iscomplexobj(csr.data)
            dtype = numpy.complex128 if self.complex else numpy.float64
            # own copies: pardiso reads these addresses in every later phase, and
            # refactorize() writes new values into them
            self.matrix = (
                numpy.array(csr.data, dtype=dtype),
                csr.indptr.astype(numpy.int64),
                csr.indices.astype(numpy.int64),
            )
            self.shape = csr.shape
            self.symmetricHalf = False if symmetricHalf is None else symmetricHalf
            self.addresses = tuple(array.ctypes.data for array in self.matrix)

    def _callPardiso(self, pardisoPhase, rhs=None, solution=None, nrhs=1):
        error = ctypes.c_int64(0)
        if self.complex:
            mattype = 13 if self.symmetric is None or not self.symmetric else 3
            mattype = ctypes.c_int64(6 if self.symmetricHalf else mattype)
        else:
            mattype = 11 if self.symmetric is None or not self.symmetric else 1
            mattype = ctypes.c_int64(2 if self.symmetricHalf else mattype)
        n = ctypes.c_int64(self.shape[0])
        a = ctypes.cast(self.addresses[0], ctypes.POINTER(ctypes.c_double))
        ia = ctypes.cast(self.addresses[1], ctypes.POINTER(ctypes.c_int64))
        ja = ctypes.cast(self.addresses[2], ctypes.POINTER(ctypes.c_int64))
        b = ctypes.cast(0 if rhs is None else rhs, ctypes.POINTER(ctypes.c_double))
        x = ctypes.cast(0 if rhs is None else solution, ctypes.POINTER(ctypes.c_double))

        self.mkl.pardiso_64(
            self.pt,  # pt        internal data structure
            ctypes.byref(
                ctypes.c_int64(1)
            ),  # maxfact   Number of matrix factorizations
            ctypes.byref(ctypes.c_int64(1)),  # mnum      Matrix index
            ctypes.byref(mattype),  # mtype     Matrix type
            ctypes.byref(
                ctypes.c_int64(pardisoPhase)
            ),  # phase     Pardiso solver phase
            ctypes.byref(n),  # n         Number of rows
            a,
            ia,
            ja,  # a, ia, ja Nonzero values, inptr, and indices
            ctypes.byref(
                ctypes.c_int64(0)
            ),  # perm      Permutation vector (not needed?)
            ctypes.byref(ctypes.c_int64(nrhs)),  # nrhs      Number of right hand sides
            self.iparm,  # iparm     Pardiso configuration parameters
            ctypes.byref(self.messageLevel),  # msglvl    Verbosity
            b,
            x,  # b, x      Right hand side(s) and Solution target(s)
            ctypes.byref(error),  # errro     Error output
        )

        if error.value != 0:
            errorMap = {
                -1: "input inconsistent",
                -2: "not enough memory",
                -3: "reordering problem",
                -4: "Zero pivot, numerical factorization or iterative refinement problem. If the "
                "error appears during the solution phase, try to change the pivoting perturba"
                "tion (iparm[9]) and also increase the number of iterative refinement steps. "
                "If it does not help, consider changing the scaling, matching and pivoting op"
                "tions (iparm[10], iparm[12], iparm[20])",
                -5: "unclassified (internal) error",
                -6: "reordering failed",
                -7: "diagonal matrix is singular",
                -8: "32-bit integer overflow problem",
                -9: "not enough memory for OOC",
                -10: "error opening OOC files",
                -11: "read/write error with OOC files",
                -12: "pardiso_64 called from 32-bit library",
                -13: "interrupted by the (user-defined) mkl_progress function",
                -15: "internal error which can appear for iparm[23]=10 and iparm[12]=1. Try switc"
                "h matching off (set iparm[12]=0 and rerun.)",
            }

            errorMsg = (
                errorMap[error.value]
                if error.value in errorMap
                else "unknown error code"
            )

            raise RuntimeError(
                f"pardiso_64 error {error.value} ({errorMsg}) in phase {pardisoPhase}."
            )

    def refactorize(self, matrix=None):  # new values on the unchanged sparsity pattern
        # pardiso keeps reading the arrays registered at analysis, so copy the new
        # values into them instead of swapping pointers (scipy path only; the mlhp
        # path updates the matrix buffer in place and passes nothing)
        if matrix is not None:
            self.matrix[0][:] = scipy.sparse.csr_matrix(matrix).data
        self._callPardiso(22)

    def cleanup(self):
        self._callPardiso(-1)

    def __call__(self, rhs, target=None):
        if isinstance(rhs, numpy.ndarray):
            # (n,) or (n, k): pardiso expects the right hand sides as contiguous
            # length-n blocks, i.e. column-major for a stacked (n, k) array
            dtype = numpy.complex128 if self.complex else numpy.float64
            b = numpy.asfortranarray(rhs, dtype=dtype)
            x = (
                numpy.empty(b.shape, dtype=dtype, order="F")
                if target is None
                else target
            )
            self._callPardiso(
                33, b.ctypes.data, x.ctypes.data, 1 if b.ndim == 1 else b.shape[1]
            )
            return x

        sol = mlhp.DoubleVector(self.shape[0]) if target is None else target
        vector_address = rhs.address if isinstance(rhs, mlhp.DoubleVector) else rhs
        target_address = sol.address if isinstance(sol, mlhp.DoubleVector) else sol

        self._callPardiso(33, vector_address, target_address)

        return sol

    solve = __call__  # drop-in for scipy factorization objects (splu et al.)

    def __del__(self):
        self.cleanup()


def pardisoSolve(matrix, vector, *, messageLevel=0, symmetric=None, mklLibrary=None):
    return pardisoFactorize(
        matrix, messageLevel=messageLevel, symmetric=symmetric, mklLibrary=mklLibrary
    )(vector)


def _checkSparseStatus(functionName, error):
    if error != 0:
        msgMap = {
            1: "SPARSE_STATUS_NOT_INITIALIZED (empty handle or matrix arrays)",
            2: "SPARSE_STATUS_ALLOC_FAILED (internal memory allocation failed)",
            3: "SPARSE_STATUS_INVALID_VALUE (input parameters contain an invalid value)",
            4: "SPARSE_STATUS_EXECUTION_FAILED (execution failed)",
            5: "SPARSE_STATUS_INTERNAL_ERROR (error in algorithm implementation occurred)",
            6: "SPARSE_STATUS_NOT_SUPPORTED (the requested operation is not supported)",
        }

        msg = msgMap[error] if error in msgMap else f" an unknown error code ({error})"

        raise RuntimeError(f"{functionName} failed with {msg}.")


class _mklCsrMatrix:
    def __init__(
        self,
        mkl,
        shape,
        indptr_address,
        indices_address,
        data_address,
        *,
        zeroBased=True,
        intType=ctypes.c_int64,
    ):
        self.ptr = ctypes.c_void_p()
        self.mkl = mkl
        a = ctypes.cast(data_address, ctypes.POINTER(ctypes.c_double))
        ia0 = ctypes.cast(indptr_address, ctypes.POINTER(intType))
        ia1 = ctypes.cast(
            indptr_address + ctypes.sizeof(intType), ctypes.POINTER(intType)
        )
        ja = ctypes.cast(indices_address, ctypes.POINTER(intType))

        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/mkl-sparse-create-csr.html
        error = mkl.mkl_sparse_d_create_csr(
            ctypes.POINTER(ctypes.c_void_p)(self.ptr),
            ctypes.c_int(0 if zeroBased else 1),
            intType(shape[0]),
            intType(shape[1]),
            ia0,
            ia1,
            ja,
            a,
        )

        _checkSparseStatus("mkl_sparse_d_create_csr", error)

    def __del__(self):
        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/mkl-sparse-destroy.html
        error = self.mkl.mkl_sparse_destroy(self.ptr)

        _checkSparseStatus("mkl_sparse_destroy", error)


class _matrixDescr(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("mode", ctypes.c_int), ("diag", ctypes.c_int)]


def _internalIlu(matrix, iluCallback, printInfo, returnMatrix, mklLibrary):

    mkl = loadMkl() if mklLibrary is None else mklLibrary

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Non-square matrix in MKL ilu.")

    if matrix.symmetricHalf:
        raise ValueError("Symmetric matrix not implemented in MKL ilu.")

    # Create new 32-bit matrix data structure
    if matrix.nnz > 2**31:
        raise ValueError("To many entries to convert to 32-bit sparse data structure.")

    ia = (ctypes.c_int32 * (matrix.shape[0] + 1))()
    ja = (ctypes.c_int32 * matrix.nnz)()

    inptr = matrix.indptr_buffer
    indices = matrix.indices_buffer

    for i in range(len(ia)):
        ia[i] = inptr[i] + 1

    for i in range(len(ja)):
        ja[i] = indices[i] + 1

    # Prepare data call mkl function
    a = ctypes.cast(matrix.data_address, ctypes.POINTER(ctypes.c_double))
    ipar = (ctypes.c_int32 * 128)()
    dpar = (ctypes.c_double * 128)()

    ib, jb, bilu = iluCallback(mkl, matrix.shape[0], a, ia, ja, ipar, dpar)

    del ia, ja, ipar, dpar

    if printInfo:
        print("MKL Incomplete LU factorization:")
        print(f"    shape              : {tuple(matrix.shape)}")
        print(f"    number of nonzeros : {ib[-1]}")
        print(f"    average bandwith   : {ib[-1] / matrix.shape[0]:.5g}")
        print(
            f"    fill ratio         : {100 * ib[-1] / matrix.shape[0] / matrix.shape[1]:.4f} %"
        )
        print(
            f"    heap memory usage  : {((len(ib) + len(jb)) * 4 + len(bilu) * 8) / 1e6:.5g} MB"
        )

    if returnMatrix:
        LU = mlhp.allocateSparseMatrix(matrix.shape, ib[-1] - 1)

        LUindptr = LU.indptr_buffer
        LUindices = LU.indices_buffer
        LUdata = LU.data_buffer

        for i in range(len(ib)):
            LUindptr[i] = ib[i] - 1

        for i in range(ib[-1] - 1):
            LUindices[i] = jb[i] - 1
            LUdata[i] = bilu[i]

        return LU

    # Create linear operator with lower triangular and upper triangular substitution
    def iluSolve(vector_address, target_address, n):
        if n != iluSolve.shape[0] or n != iluSolve.shape[1]:
            raise ValueError("Inconsistent size in ilu substitution.")

        # See mkl_spblas.h
        SPARSE_OPERATION_NON_TRANSPOSE = 10
        SPARSE_MATRIX_TYPE_TRIANGULAR = 23
        SPARSE_FILL_MODE_LOWER = 40
        SPARSE_FILL_MODE_UPPER = 41
        SPARSE_DIAG_NON_UNIT = 50
        SPARSE_DIAG_UNIT = 51

        # Prepare triangular solve function arguments
        vector = ctypes.cast(vector_address, ctypes.POINTER(ctypes.c_double))
        tmp = ctypes.cast(iluSolve.tmp.address, ctypes.POINTER(ctypes.c_double))
        target = ctypes.cast(target_address, ctypes.POINTER(ctypes.c_double))

        # Lower triangular substitution with unit-diagonal, the SciPy equivalent of:
        # tmp = scipy.linalg.solve_triangular(LU, vector, lower=True, unit_diagonal=True)
        # Where LU could be, e.g.: LU, P = scipy.linalg.lu_factor(matrix.todense())
        descr = _matrixDescr(
            type=SPARSE_MATRIX_TYPE_TRIANGULAR,
            mode=SPARSE_FILL_MODE_LOWER,
            diag=SPARSE_DIAG_UNIT,
        )

        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/mkl-sparse-trsv.html
        error = iluSolve.mkl.mkl_sparse_d_trsv(
            ctypes.c_int(SPARSE_OPERATION_NON_TRANSPOSE),
            ctypes.c_double(1.0),
            iluSolve.A.ptr,
            descr,
            vector,
            tmp,
        )

        _checkSparseStatus("mkl_sparse_d_trsv", error)

        # Upper triangular substitution, the SciPy equivalent of:
        # scipy.linalg.solve_triangular(LU, tmp, lower=False, unit_diagonal=False)
        descr = _matrixDescr(
            type=SPARSE_MATRIX_TYPE_TRIANGULAR,
            mode=SPARSE_FILL_MODE_UPPER,
            diag=SPARSE_DIAG_NON_UNIT,
        )

        error = iluSolve.mkl.mkl_sparse_d_trsv(
            ctypes.c_int(SPARSE_OPERATION_NON_TRANSPOSE),
            ctypes.c_double(1.0),
            iluSolve.A.ptr,
            descr,
            tmp,
            target,
        )

        _checkSparseStatus("mkl_sparse_d_trsv", error)

    iluSolve.shape = matrix.shape
    iluSolve.mkl = mkl
    iluSolve.tmp = mlhp.DoubleVector(matrix.shape[0])
    iluSolve.data = (ib, jb, bilu)

    iluSolve.A = _mklCsrMatrix(
        mkl=mkl,
        shape=matrix.shape,
        indptr_address=ctypes.cast(ib, ctypes.c_void_p).value,
        indices_address=ctypes.cast(jb, ctypes.c_void_p).value,
        data_address=ctypes.cast(bilu, ctypes.c_void_p).value,
        zeroBased=False,
        intType=ctypes.c_int32,
    )

    return mlhp.linearOperator_address(iluSolve)


def ilu0(matrix, *, printInfo=False, returnMatrix=False, mklLibrary=None):
    def ilu0Callback(mkl, n, a, ia, ja, ipar, dpar):

        bilu0 = (ctypes.c_double * (ia[n] - 1))()
        ierr = ctypes.c_int32(0)
        n = ctypes.c_int32(n)

        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2025-1/dcsrilu0.html
        mkl.dcsrilu0(
            ctypes.byref(n),  # Number of rows
            a,
            ia,
            ja,
            bilu0,  # Nonzero source values, inptr, indices, and target values
            ipar,
            dpar,  # Integer and double precision set description
            ctypes.byref(ierr),  # Error output
        )

        if ierr.value != 0:
            errorMap = {
                -101: "At least one diagonal element is omitted from the matrix in CSR3 format (see S"
                "parse Matrix Storage Format).",
                -102: "Matrix contains a diagonal element with the value of zero.",
                -103: "Matrix contains a diagonal element which is so small that it could cause an ov"
                "erflow, or that it would cause a bad approximation to ILU0.",
                -104: "Memory is insufficient for the internal work array.",
                -105: "Input matrix size n is less than or equal to 0.",
                -106: "Column indices ja are not in the ascending order.",
            }

            errorMsg = (
                errorMap[ierr.value] if ierr.value in errorMap else "unknown error code"
            )

            raise RuntimeError(f"dcsrilu0 error {ierr.value} ({errorMsg}).")

        return ia, ja, bilu0

    return _internalIlu(matrix, ilu0Callback, printInfo, returnMatrix, mklLibrary)


def ilut(
    matrix,
    drop_tol=1e-4,
    fill_factor=10,
    *,
    printInfo=False,
    returnMatrix=False,
    mklLibrary=None,
):
    def ilutCallback(mkl, n, a, ia, ja, ipar, dpar):
        maxfil, tol = ilutCallback.maxfil, ilutCallback.tol
        dpar[30] = tol
        ipar[30] = 1
        nnz = (2 * maxfil + 1) * n - maxfil * (maxfil + 1) + 1
        ib = (ctypes.c_int32 * (n + 1))()
        jb = (ctypes.c_int32 * nnz)()
        b = (ctypes.c_double * nnz)()
        ierr = ctypes.c_int32(0)
        n = ctypes.c_int32(n)
        tol = ctypes.c_double(tol)
        maxfil = ctypes.c_int(maxfil)

        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2025-1/dcsrilut.html
        mkl.dcsrilut(
            ctypes.byref(n),  # Number of rows
            a,
            ia,
            ja,  # Nonzero source values, inptr, and indices
            b,
            ib,
            jb,  # Nonzero target for values, inptr, and indices
            ctypes.byref(tol),  # Tolerance above which to keep entries
            ctypes.byref(
                maxfil
            ),  # Half of the bandwidth (for each row: bandwidth <= 2 * maxfill + 1)
            ipar,
            dpar,  # Integer and double precision set description
            ctypes.byref(ierr),  # Error output
        )

        if ierr.value != 0:
            errorMap = {
                -101: "the number of elements in some matrix row specified in the sparse format is equ"
                "al to or less than 0.",
                -102: "value of the computed diagonal element is less than the product of the given to"
                "lerance and the current matrix row norm, and it cannot be replaced as ipar(31)="
                "0.",
                -103: "element ia(i + 1) is less than or equal to the element ia(i) (see Sparse Matrix"
                " Storage Format).",
                -104: "memory is insufficient for the internal work arrays.",
                -105: "input value of maxfil is less than 0.",
                -106: "size n of the input matrix is less than 0.",
                -107: "element of the array ja is less than 1, or greater than n (see Sparse Matrix St"
                "orage Format).",
                101: "value of maxfil is greater than or equal to n. The calculation is performed wit"
                "h the value of maxfil set to (n-1).",
                102: "value of tol is less than 0. The calculation is performed with the value of the"
                " parameter set to (-tol)",
                103: "absolute value of tol is greater than value of dpar(31); it can result in insta"
                "bility of the calculation.",
                104: "value of dpar(31) is equal to 0. It can cause calculations to fail.",
            }

            errorMsg = (
                errorMap[ierr.value] if ierr.value in errorMap else "unknown error code"
            )

            raise RuntimeError(f"dcsrilut error {ierr.value} ({errorMsg}).")

        return ib, jb, b

    maxfil = (
        int((fill_factor * (matrix.nnz - matrix.shape[0])) / (2 * matrix.shape[0])) + 1
    )

    ilutCallback.maxfil = min(maxfil, matrix.shape[1] - 1)
    ilutCallback.tol = drop_tol

    return _internalIlu(matrix, ilutCallback, printInfo, returnMatrix, mklLibrary)


def linearOperator(matrix, *, symmetric=False, mklLibrary=None):
    mkl = loadMkl() if mklLibrary is None else mklLibrary

    def SpMV(vector_address, target_address, n):
        if n != SpMV.matrix.shape[0] or n != SpMV.matrix.shape[1]:
            raise ValueError("Inconsistent size in ilu substitution.")

        operation = ctypes.c_int(10)  # transpose: 11
        alpha = ctypes.c_double(1.0)
        beta = ctypes.c_double(1.0)

        vector = ctypes.cast(vector_address, ctypes.POINTER(ctypes.c_double))
        target = ctypes.cast(target_address, ctypes.POINTER(ctypes.c_double))

        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-1/mkl-sparse-mv.html
        error = SpMV.mkl.mkl_sparse_d_mv(
            operation, alpha, SpMV.A.ptr, SpMV.descr, vector, beta, target
        )

        _checkSparseStatus("mkl_sparse_d_mv", error)

    # Copy to 32-bit data structure
    if matrix.nnz > 2**31:
        raise ValueError("To many entries to convert to 32-bit sparse data structure.")

    ia = (ctypes.c_int32 * (matrix.shape[0] + 1))()
    ja = (ctypes.c_int32 * matrix.nnz)()

    indptr, indices = matrix.indptr_buffer, matrix.indices_buffer

    for i in range(matrix.shape[0] + 1):
        ia[i] = indptr[i]

    for i in range(matrix.nnz):
        ja[i] = indices[i]

    # Prepare mkl data
    SPARSE_MATRIX_TYPE_GENERAL = 20
    SPARSE_MATRIX_TYPE_SYMMETRIC = 21
    SPARSE_FILL_MODE_UPPER = 41
    SPARSE_FILL_MODE_FULL = 42
    SPARSE_DIAG_NON_UNIT = 50

    type = (
        SPARSE_MATRIX_TYPE_SYMMETRIC
        if matrix.symmetricHalf or symmetric
        else SPARSE_MATRIX_TYPE_GENERAL
    )
    mode = SPARSE_FILL_MODE_UPPER if matrix.symmetricHalf else SPARSE_FILL_MODE_FULL

    SpMV.mkl = mkl
    SpMV.matrix = matrix
    SpMV.data = (ia, ja)
    SpMV.descr = _matrixDescr(type=type, mode=mode, diag=SPARSE_DIAG_NON_UNIT)

    SpMV.A = _mklCsrMatrix(
        mkl=mkl,
        shape=matrix.shape,
        indptr_address=ctypes.cast(ia, ctypes.c_void_p).value,
        indices_address=ctypes.cast(ja, ctypes.c_void_p).value,
        data_address=matrix.data_address,
        zeroBased=True,
        intType=ctypes.c_int32,
    )

    return mlhp.linearOperator_address(SpMV)


# import mlhp, numpy
# mesh = mlhp.makeRefinedGrid([4, 3, 5], [3, 2, 4])
# basis = mlhp.makeHpTrunkSpace(mesh, 2, nfields=mesh.ndim)
# dirichlet = mlhp.integrateDirichletDofs(mlhp.vectorField(mesh.ndim, [0.0]*mesh.ndim), basis, [0])
# matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
# vector = mlhp.allocateRhsVector(matrix)
# kinematics = mlhp.smallStrainKinematics(mesh.ndim)
# material = mlhp.isotropicElasticMaterial(mlhp.scalarField(mesh.ndim, 1.0), mlhp.scalarField(mesh.ndim, 0.3))
# integrand = mlhp.staticDomainIntegrand(kinematics, material, mlhp.vectorField(mesh.ndim, [1.0] * mesh.ndim))
# mlhp.integrateOnDomain(basis, integrand, [matrix, vector], dirichletDofs=dirichlet)
#
# operator = linearOperator(matrix)
# print(operator(vector).tolist())

##mlhp.norm(mlhp.add(mkl.pardisoSolve(matrix, vector), mlhp.makeCGSolver(rtol=1e-14)(matrix, vector), -1.0))
# print(numpy.round(matrix.todense(), 2))
#
# import mklwrapper as mkl
# LU = mkl.ilut(matrix, returnMatrix=True, fill_factor=1e10, drop_tol=1e-200).todense()
# LU, perm = scipy.linalg.lu_factor(matrix.todense())
# tmp = scipy.linalg.solve_triangular(LU, vector, lower=True, unit_diagonal=True)
# result = scipy.linalg.solve_triangular(LU, tmp, lower=False, unit_diagonal=False)
# print(scipy.linalg.norm(mkl.pardisoSolve(matrix, vector) - result))
# LUop = mkl.ilut(matrix, returnMatrix=False, fill_factor=1e10, drop_tol=1e-200)
# print(scipy.linalg.norm(mkl.pardisoSolve(matrix, vector) - numpy.array(LUop(vector).tolist()))/scipy.linalg.norm(mkl.pardisoSolve(matrix, vector)))

if __name__ == "__main__":
    print("Testing mkl...", flush=True)

    shape = [6, 6]
    umatrix = mlhp.allocateSparseMatrix(shape, shape[0] * shape[1])
    smatrix = mlhp.allocateSparseMatrix(
        shape, (shape[0] * (shape[0] + 1)) // 2, symmetric=True
    )
    rhs = mlhp.allocateRhsVector(umatrix)

    uindex, sindex = 0, 0
    for i in range(shape[0]):
        rhs[i] = (
            0.7 * (shape[0] - i) * i / (shape[0] ** 2 / 4) + 1.2 * i / shape[0] + 3.1
        )

        umatrix.indptr_buffer[i] = uindex
        smatrix.indptr_buffer[i] = sindex

        for j in range(shape[1]):
            umatrix.indices_buffer[uindex] = j
            umatrix.data_buffer[uindex] = 2.3 / (abs(i - j) + 1)

            if j >= i:
                smatrix.indices_buffer[sindex] = j
                smatrix.data_buffer[sindex] = umatrix.data_buffer[uindex]

                sindex += 1
            uindex += 1

        umatrix.indptr_buffer[i + 1] = uindex
        smatrix.indptr_buffer[i + 1] = sindex

    expected = [
        0.42655929910747536,
        0.4971564661821119,
        0.5773572916154234,
        0.6412062015071258,
        0.7093083642299071,
        1.068437405980519,
    ]

    computed1 = pardisoSolve(umatrix, rhs)
    computed2 = pardisoSolve(umatrix, rhs, symmetric=True)
    computed3 = pardisoSolve(smatrix, rhs)
    computed4 = pardisoSolve(smatrix, rhs, symmetric=True)  # Shouldn't do anything
    computed5 = ilu0(umatrix)(rhs)
    computed6 = ilut(umatrix)(rhs)

    for computed, name in [
        (computed1, "Unsymmetric pardiso (general)"),
        (computed2, "Unsymmetric pardiso (symmetric)"),
        (computed3, "Symmetric half pardiso"),
        (computed4, "Symmetric half pardiso (symmetric)"),
        (computed5, "MKL ilu0"),
        (computed6, "MKL ilut"),
    ]:
        if sum((c - e) ** 2 for c, e in zip(computed, expected)) > 1e-12**2:
            raise ValueError(f"{name} solution incorrect.")
        else:
            print(f"{name} solution correct.", flush=True)

    solution = mlhp.DoubleVector(expected)

    mklSpuMV1 = linearOperator(umatrix, symmetric=False)
    computed1 = mklSpuMV1(solution)
    computed2 = linearOperator(umatrix, symmetric=True)(solution)
    computed3 = linearOperator(smatrix, symmetric=False)(solution)
    computed4 = linearOperator(smatrix, symmetric=True)(solution)

    for computed, name in [
        (computed1, "Unsymmetric SpMV (general)"),
        (computed2, "Unsymmetric SpMV (symmetric)"),
        (computed3, "Symmetric SpMV (general)"),
        (computed4, "Symmetric SpMV (symmetric)"),
    ]:
        if sum((c - e) ** 2 for c, e in zip(computed, rhs)) > 1e-12**2:
            raise ValueError(f"{name} incorrect.")
        else:
            print(f"{name} correct.", flush=True)

    print("Testing scipy interface...", flush=True)

    rng = numpy.random.default_rng(0)
    n = 60
    B1 = scipy.sparse.random(n, n, density=0.1, random_state=1)
    B2 = scipy.sparse.random(n, n, density=0.1, random_state=2)
    S = (B1 + B1.T + 10.0 * scipy.sparse.eye(n)) + 1j * (
        B2 + B2.T + scipy.sparse.eye(n)
    )
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    for kwargs in [dict(), dict(symmetric=True)]:
        factor = pardisoFactorize(S.tocsr(), **kwargs)
        if numpy.abs(S @ factor(b) - b).max() > 1e-10:
            raise ValueError(f"Complex scipy pardiso {kwargs} solution incorrect.")
        factor.refactorize(2.0 * S.tocsr())
        if numpy.abs(2.0 * (S @ factor(b)) - b).max() > 1e-10:
            raise ValueError(f"Complex scipy pardiso {kwargs} refactorize incorrect.")
        print(f"Complex scipy pardiso {kwargs} correct.", flush=True)

    Areal = (B1 + B1.T + 10.0 * scipy.sparse.eye(n)).tocsr()
    if numpy.abs(Areal @ pardisoFactorize(Areal)(b.real) - b.real).max() > 1e-10:
        raise ValueError("Real scipy pardiso solution incorrect.")
    print("Real scipy pardiso correct.", flush=True)

    B = numpy.column_stack([b, 2.0 * b.conj(), b * 1j])
    if numpy.abs(S @ pardisoFactorize(S.tocsr())(B) - B).max() > 1e-10:
        raise ValueError("Multi-rhs pardiso solution incorrect.")
    print("Multi-rhs pardiso correct.", flush=True)
