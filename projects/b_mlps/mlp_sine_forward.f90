program mlp_sine_forward
    implicit none

    type :: dense
        real(8), allocatable :: W(:, :), b(:)
    end type

    integer, parameter :: LAYERS(5) = [1, 24, 24, 24, 1]
    type(dense) :: net(size(LAYERS) - 1)
    real(8), allocatable :: a(:)
    integer :: i, unit

    ! ---------------------------------- load weights ---------------------------------
    open(newunit=unit, file="../../models/mlp_sine_weights.txt")
    do i = 1, size(net)
        allocate(net(i)%W(LAYERS(i + 1), LAYERS(i)))
        allocate(net(i)%b(LAYERS(i + 1)))
        read(unit, *) net(i)%W
        read(unit, *) net(i)%b
    end do
    close(unit)

    ! ---------------------------------- forward pass ---------------------------------
    a = [0.25d0]
    do i = 1, size(net) - 1
        a = tanh(matmul(net(i)%W, a) + net(i)%b)
    end do
    a = matmul(net(size(net))%W, a) + net(size(net))%b
    print "(a, f9.6)", "prediction ", a(1)
end program
