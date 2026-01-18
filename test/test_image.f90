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
            new_unittest("flip_v_twice_identity", test_flip_v_twice_identity), &
            new_unittest("rotation_matrix_identity", test_rotation_matrix_identity), &
            new_unittest("rotation_matrix_90", test_rotation_matrix_90), &
            new_unittest("translation_matrix_values", test_translation_matrix_values), &
            new_unittest("affine_identity", test_affine_identity), &
            new_unittest("affine_translate", test_affine_translate), &
            new_unittest("affine_rotate_180", test_affine_rotate_180), &
            new_unittest("affine_rotate_360", test_affine_rotate_360), &
            new_unittest("affine_bilinear", test_affine_bilinear), &
            new_unittest("affine_edge_clamp", test_affine_edge_clamp), &
            new_unittest("affine_preserves_channels", test_affine_preserves_channels), &
            new_unittest("scale_matrix_identity", test_scale_matrix_identity), &
            new_unittest("scale_matrix_values", test_scale_matrix_values), &
            new_unittest("affine_zoom_in", test_affine_zoom_in), &
            new_unittest("affine_zoom_out", test_affine_zoom_out) &
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


    subroutine test_rotation_matrix_identity(error)
        type(error_type), allocatable, intent(out) :: error
        real :: M(3,3), identity(3,3)

        M = rotation_matrix(0.0, 0.0, 0.0)  ! 0 degree rotation

        identity = reshape([1.0, 0.0, 0.0, &
                            0.0, 1.0, 0.0, &
                            0.0, 0.0, 1.0], [3, 3])

        call check(error, all(abs(M - identity) < 1.0e-6), "0 degree rotation should be identity")
    end subroutine


    subroutine test_rotation_matrix_90(error)
        type(error_type), allocatable, intent(out) :: error
        real :: M(3,3)

        M = rotation_matrix(0.0, 0.0, 90.0)  ! 90 degrees around origin

        ! cos(90) = 0, sin(90) = 1
        call check(error, abs(M(1,1)) < 1.0e-6, "M(1,1) should be ~0")
        if (allocated(error)) return
        call check(error, abs(M(1,2) + 1.0) < 1.0e-6, "M(1,2) should be ~-1")
        if (allocated(error)) return
        call check(error, abs(M(2,1) - 1.0) < 1.0e-6, "M(2,1) should be ~1")
        if (allocated(error)) return
        call check(error, abs(M(2,2)) < 1.0e-6, "M(2,2) should be ~0")
    end subroutine


    subroutine test_translation_matrix_values(error)
        type(error_type), allocatable, intent(out) :: error
        real :: M(3,3)

        M = translation_matrix(5.0, 10.0)

        call check(error, abs(M(1,1) - 1.0) < 1.0e-6, "M(1,1) should be 1")
        if (allocated(error)) return
        call check(error, abs(M(1,3) - 5.0) < 1.0e-6, "tx should be 5")
        if (allocated(error)) return
        call check(error, abs(M(2,3) - 10.0) < 1.0e-6, "ty should be 10")
    end subroutine


    subroutine test_affine_identity(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        real :: M(3,3)
        integer :: c, x, y

        allocate(img(3, 5, 5))
        do c = 1, 3
            do x = 1, 5
                do y = 1, 5
                    img(c, x, y) = real(c * 100 + x * 10 + y) / 1000.0
                end do
            end do
        end do

        M = reshape([1.0, 0.0, 0.0, &
                     0.0, 1.0, 0.0, &
                     0.0, 0.0, 1.0], [3, 3])

        result = affine_transform(img, M)

        call check(error, all(abs(result - img) < 1.0e-5), "identity transform should preserve image")
    end subroutine


    subroutine test_affine_translate(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        real :: M(3,3)

        allocate(img(1, 5, 5))
        img = 0.0
        img(1, 3, 3) = 1.0  ! center pixel

        M = translation_matrix(1.0, 0.0)  ! shift right by 1

        result = affine_transform(img, M)

        ! pixel at (3,3) should now be at (4,3)
        call check(error, result(1, 4, 3) > 0.5, "pixel should shift right")
    end subroutine


    subroutine test_affine_rotate_180(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        real :: M(3,3)

        allocate(img(1, 5, 5))
        img = 0.0
        img(1, 2, 2) = 1.0  ! off-center pixel

        ! rotate 180 around center (3,3)
        M = rotation_matrix(3.0, 3.0, 180.0)
        result = affine_transform(img, M)

        ! pixel at (2,2) should move to (4,4)
        call check(error, result(1, 4, 4) > 0.5, "pixel should rotate to opposite corner")
        if (allocated(error)) return
        call check(error, result(1, 2, 2) < 0.1, "original position should be empty")
    end subroutine


    subroutine test_affine_rotate_360(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        real :: M(3,3)
        integer :: c, x, y

        allocate(img(1, 7, 7))
        do x = 1, 7
            do y = 1, 7
                img(1, x, y) = real(x * 10 + y) / 100.0
            end do
        end do

        ! rotate 360 around center
        M = rotation_matrix(4.0, 4.0, 360.0)
        result = affine_transform(img, M)

        ! should be ~identity (within interpolation tolerance)
        call check(error, all(abs(result - img) < 0.02), "360 rotation should be ~identity")
    end subroutine


    subroutine test_affine_bilinear(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        real :: M(3,3)

        allocate(img(1, 4, 4))
        img = 0.0
        img(1, 2, 2) = 1.0

        ! translate by 0.5 pixels - should spread value via bilinear
        M = translation_matrix(0.5, 0.0)
        result = affine_transform(img, M)

        ! value should be split between (2,2) and (3,2)
        call check(error, result(1, 2, 2) > 0.3 .and. result(1, 2, 2) < 0.7, &
                   "bilinear should split value at source")
        if (allocated(error)) return
        call check(error, result(1, 3, 2) > 0.3 .and. result(1, 3, 2) < 0.7, &
                   "bilinear should split value at dest")
    end subroutine


    subroutine test_affine_edge_clamp(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        real :: M(3,3)

        allocate(img(1, 5, 5))
        img = 0.0
        img(1, 5, :) = 1.0  ! right edge is white

        ! translate with tx=-2 shifts content left, so right side samples beyond bounds
        M = translation_matrix(-2.0, 0.0)
        result = affine_transform(img, M)

        ! output pixel (5,3) samples from input (7,3) which clamps to (5,3) = white
        call check(error, result(1, 5, 3) > 0.9, "edge should be clamped/replicated")
    end subroutine


    subroutine test_affine_preserves_channels(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        real :: M(3,3)

        allocate(img(3, 5, 5))
        img(1, :, :) = 0.2  ! R
        img(2, :, :) = 0.5  ! G
        img(3, :, :) = 0.8  ! B

        M = translation_matrix(1.0, 1.0)
        result = affine_transform(img, M)

        ! check center pixel has correct channel values
        call check(error, abs(result(1, 3, 3) - 0.2) < 0.01, "R channel preserved")
        if (allocated(error)) return
        call check(error, abs(result(2, 3, 3) - 0.5) < 0.01, "G channel preserved")
        if (allocated(error)) return
        call check(error, abs(result(3, 3, 3) - 0.8) < 0.01, "B channel preserved")
    end subroutine


    subroutine test_scale_matrix_identity(error)
        type(error_type), allocatable, intent(out) :: error
        real :: M(3,3), identity(3,3)

        M = scale_matrix(0.0, 0.0, 1.0, 1.0)  ! scale by 1 = identity

        identity = reshape([1.0, 0.0, 0.0, &
                            0.0, 1.0, 0.0, &
                            0.0, 0.0, 1.0], [3, 3])

        call check(error, all(abs(M - identity) < 1.0e-6), "scale by 1 should be identity")
    end subroutine


    subroutine test_scale_matrix_values(error)
        type(error_type), allocatable, intent(out) :: error
        real :: M(3,3)

        M = scale_matrix(0.0, 0.0, 2.0, 3.0)  ! scale around origin

        call check(error, abs(M(1,1) - 2.0) < 1.0e-6, "sx should be 2")
        if (allocated(error)) return
        call check(error, abs(M(2,2) - 3.0) < 1.0e-6, "sy should be 3")
        if (allocated(error)) return
        call check(error, abs(M(1,3)) < 1.0e-6, "tx should be 0 for origin center")
        if (allocated(error)) return
        call check(error, abs(M(2,3)) < 1.0e-6, "ty should be 0 for origin center")
    end subroutine


    subroutine test_affine_zoom_in(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        real :: M(3,3)

        allocate(img(1, 5, 5))
        img = 0.0
        img(1, 3, 3) = 1.0  ! center pixel

        ! zoom in 2x around center - center should stay at center
        M = scale_matrix(3.0, 3.0, 2.0, 2.0)
        result = affine_transform(img, M)

        call check(error, result(1, 3, 3) > 0.9, "center pixel should remain after zoom in")
    end subroutine


    subroutine test_affine_zoom_out(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        real :: M(3,3)

        allocate(img(1, 5, 5))
        img = 1.0  ! all white
        img(1, 3, 3) = 0.0  ! black center pixel

        ! zoom out 0.5x around center
        M = scale_matrix(3.0, 3.0, 0.5, 0.5)
        result = affine_transform(img, M)

        ! center should still be dark (sampling from center)
        call check(error, result(1, 3, 3) < 0.1, "center should remain dark after zoom out")
        if (allocated(error)) return
        ! corners should sample from outside bounds (clamped to edge = white)
        call check(error, result(1, 1, 1) > 0.9, "corner should be white after zoom out")
    end subroutine

end module test_image
