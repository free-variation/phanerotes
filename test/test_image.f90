module test_image
    use testdrive, only: new_unittest, unittest_type, error_type, check
    use image
    implicit none
    private

    public :: collect_image_tests

contains

    subroutine collect_image_tests(testsuite)
        type(unittest_type), allocatable, intent(out) :: testsuite(:)

        testsuite = [ &
            new_unittest("transpose_swaps_dims", test_transpose_swaps_dims), &
            new_unittest("transpose_pixel_position", test_transpose_pixel_position), &
            new_unittest("transpose_twice_identity", test_transpose_twice_identity), &
            new_unittest("flip_h_swaps_columns", test_flip_h_swaps_columns), &
            new_unittest("flip_h_twice_identity", test_flip_h_twice_identity), &
            new_unittest("flip_v_swaps_rows", test_flip_v_swaps_rows), &
            new_unittest("flip_v_twice_identity", test_flip_v_twice_identity) &
        ]
    end subroutine


    subroutine test_transpose_swaps_dims(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)

        allocate(img(3, 4, 5))  ! 3 channels, 4 wide, 5 tall
        img = 0.0

        result = transpose_image(img)

        call check(error, size(result, 1) == 3, "channels should be preserved")
        if (allocated(error)) return
        call check(error, size(result, 2) == 5, "width should become old height")
        if (allocated(error)) return
        call check(error, size(result, 3) == 4, "height should become old width")
    end subroutine


    subroutine test_transpose_pixel_position(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)

        allocate(img(1, 3, 2))
        img = 0.0
        img(1, 2, 1) = 1.0  ! mark position (x=2, y=1)

        result = transpose_image(img)

        call check(error, result(1, 1, 2) == 1.0, "pixel should move to (x=1, y=2)")
    end subroutine


    subroutine test_transpose_twice_identity(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        integer :: c, x, y

        allocate(img(3, 4, 5))
        do c = 1, 3
            do x = 1, 4
                do y = 1, 5
                    img(c, x, y) = real(c * 100 + x * 10 + y)
                end do
            end do
        end do

        result = transpose_image(transpose_image(img))

        call check(error, all(abs(result - img) < 1.0e-6), "double transpose should be identity")
    end subroutine


    subroutine test_flip_h_swaps_columns(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)

        allocate(img(1, 3, 2))
        img(1, 1, :) = 0.1
        img(1, 2, :) = 0.5
        img(1, 3, :) = 0.9

        result = flip_image_horizontal(img)

        call check(error, all(abs(result(1, 1, :) - 0.9) < 1.0e-6), "first col should be last")
        if (allocated(error)) return
        call check(error, all(abs(result(1, 3, :) - 0.1) < 1.0e-6), "last col should be first")
    end subroutine


    subroutine test_flip_h_twice_identity(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        integer :: c, x, y

        allocate(img(3, 4, 5))
        do c = 1, 3
            do x = 1, 4
                do y = 1, 5
                    img(c, x, y) = real(c * 100 + x * 10 + y)
                end do
            end do
        end do

        result = flip_image_horizontal(flip_image_horizontal(img))

        call check(error, all(abs(result - img) < 1.0e-6), "double flip_h should be identity")
    end subroutine


    subroutine test_flip_v_swaps_rows(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)

        allocate(img(1, 2, 3))
        img(1, :, 1) = 0.1
        img(1, :, 2) = 0.5
        img(1, :, 3) = 0.9

        result = flip_image_vertical(img)

        call check(error, all(abs(result(1, :, 1) - 0.9) < 1.0e-6), "first row should be last")
        if (allocated(error)) return
        call check(error, all(abs(result(1, :, 3) - 0.1) < 1.0e-6), "last row should be first")
    end subroutine


    subroutine test_flip_v_twice_identity(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        integer :: c, x, y

        allocate(img(3, 4, 5))
        do c = 1, 3
            do x = 1, 4
                do y = 1, 5
                    img(c, x, y) = real(c * 100 + x * 10 + y)
                end do
            end do
        end do

        result = flip_image_vertical(flip_image_vertical(img))

        call check(error, all(abs(result - img) < 1.0e-6), "double flip_v should be identity")
    end subroutine

end module test_image
