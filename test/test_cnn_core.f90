program test_cnn_core
    use cnn_core
    implicit none

    ! im2col tests
    call test_im2col_basic()
    call test_im2col_padding()

    ! col2im tests
    call test_col2im_roundtrip()
    call test_col2im_accumulation()

    ! Combined tests
    call test_full_configurations()

contains

    subroutine test_im2col_basic()
        real, allocatable :: input(:,:,:)
        real, allocatable :: result(:,:)
        real :: expected(4, 4)
        integer :: i, j, errors

        ! 1 channel, 4x4 input, 2x2 kernel, stride 2, no padding
        allocate(input(1, 4, 4))
        do j = 1, 4
            do i = 1, 4
                input(1, i, j) = real((j - 1) * 4 + i)
            end do
        end do

        result = im2col(input, 2, 2, 2, 0)

        expected(:, 1) = [1.0, 2.0, 5.0, 6.0]
        expected(:, 2) = [3.0, 4.0, 7.0, 8.0]
        expected(:, 3) = [9.0, 10.0, 13.0, 14.0]
        expected(:, 4) = [11.0, 12.0, 15.0, 16.0]

        if (size(result, 1) /= 4 .or. size(result, 2) /= 4) then
            print *, "FAIL im2col_basic: wrong dimensions"
            error stop
        end if

        errors = 0
        do j = 1, 4
            do i = 1, 4
                if (abs(result(i, j) - expected(i, j)) > 1e-6) then
                    errors = errors + 1
                end if
            end do
        end do

        if (errors == 0) then
            print *, "PASS: im2col basic"
        else
            print *, "FAIL im2col_basic:", errors, "errors"
            error stop
        end if

        deallocate(input, result)
    end subroutine

    subroutine test_im2col_padding()
        real, allocatable :: input(:,:,:)
        real, allocatable :: result(:,:)

        allocate(input(1, 3, 3))
        input(1, :, :) = reshape([1.0, 2.0, 3.0, &
                                  4.0, 5.0, 6.0, &
                                  7.0, 8.0, 9.0], [3, 3])

        result = im2col(input, 2, 2, 1, 1)

        if (size(result, 1) /= 4 .or. size(result, 2) /= 16) then
            print *, "FAIL im2col_padding: wrong dimensions"
            error stop
        end if

        ! Corner patch should have zeros from padding: [0, 0, 0, 1]
        if (abs(result(1, 1)) > 1e-6 .or. &
            abs(result(2, 1)) > 1e-6 .or. &
            abs(result(3, 1)) > 1e-6 .or. &
            abs(result(4, 1) - 1.0) > 1e-6) then
            print *, "FAIL im2col_padding: corner values wrong"
            error stop
        end if

        print *, "PASS: im2col padding"
        deallocate(input, result)
    end subroutine

    subroutine test_col2im_roundtrip()
        real, allocatable :: input(:,:,:)
        real, allocatable :: col_matrix(:,:)
        real, allocatable :: reconstructed(:,:,:)
        integer :: i, j, errors

        allocate(input(1, 4, 4))
        do j = 1, 4
            do i = 1, 4
                input(1, i, j) = real((j - 1) * 4 + i)
            end do
        end do

        col_matrix = im2col(input, 2, 2, 2, 0)
        reconstructed = col2im(col_matrix, 1, 4, 4, 2, 2, 2, 0)

        errors = 0
        do j = 1, 4
            do i = 1, 4
                if (abs(reconstructed(1, i, j) - input(1, i, j)) > 1e-6) then
                    errors = errors + 1
                end if
            end do
        end do

        if (errors == 0) then
            print *, "PASS: col2im roundtrip"
        else
            print *, "FAIL col2im_roundtrip:", errors, "errors"
            error stop
        end if

        deallocate(input, col_matrix, reconstructed)
    end subroutine

    subroutine test_col2im_accumulation()
        real, allocatable :: col_matrix(:,:)
        real, allocatable :: result(:,:,:)

        ! 3x3 output, 2x2 kernel, stride 1 -> 4 overlapping patches
        allocate(col_matrix(4, 4))
        col_matrix = 1.0

        result = col2im(col_matrix, 1, 3, 3, 2, 2, 1, 0)

        ! Center (2,2) should accumulate 4x, edges 2x, corners 1x
        if (abs(result(1, 1, 1) - 1.0) > 1e-6 .or. &
            abs(result(1, 2, 1) - 2.0) > 1e-6 .or. &
            abs(result(1, 2, 2) - 4.0) > 1e-6 .or. &
            abs(result(1, 3, 3) - 1.0) > 1e-6) then
            print *, "FAIL col2im_accumulation: wrong values"
            error stop
        end if

        print *, "PASS: col2im accumulation"
        deallocate(col_matrix, result)
    end subroutine

    subroutine test_full_configurations()
        integer :: passed, total

        passed = 0
        total = 7

        if (run_config(3, 8, 8, 3, 3, 1, 1))   passed = passed + 1
        if (run_config(1, 10, 10, 5, 5, 2, 0)) passed = passed + 1
        if (run_config(64, 16, 16, 3, 3, 1, 1)) passed = passed + 1
        if (run_config(3, 12, 8, 3, 3, 1, 0))  passed = passed + 1
        if (run_config(1, 8, 8, 3, 5, 1, 0))   passed = passed + 1
        if (run_config(3, 32, 32, 4, 4, 4, 0)) passed = passed + 1
        if (run_config(1, 4, 4, 3, 3, 1, 2))   passed = passed + 1

        if (passed == total) then
            print *, "PASS: all", total, "configurations"
        else
            print *, "FAIL:", total - passed, "of", total, "failed"
            error stop
        end if
    end subroutine

    function run_config(nc, w, h, kw, kh, stride, pad) result(ok)
        integer, intent(in) :: nc, w, h, kw, kh, stride, pad
        logical :: ok

        real, allocatable :: input(:,:,:), col(:,:), output(:,:,:)
        integer :: i, j, c, out_w, out_h
        real :: max_err

        ok = .false.

        allocate(input(nc, w, h))
        do c = 1, nc
            do j = 1, h
                do i = 1, w
                    input(c, i, j) = real(c * 1000 + j * 100 + i)
                end do
            end do
        end do

        col = im2col(input, kw, kh, stride, pad)

        out_w = (w + 2*pad - kw) / stride + 1
        out_h = (h + 2*pad - kh) / stride + 1

        if (size(col, 1) /= nc * kw * kh .or. size(col, 2) /= out_w * out_h) then
            print *, "  FAIL config: im2col dims"
            return
        end if

        output = col2im(col, nc, w, h, kw, kh, stride, pad)

        if (size(output, 1) /= nc .or. size(output, 2) /= w .or. size(output, 3) /= h) then
            print *, "  FAIL config: col2im dims"
            return
        end if

        if (stride >= kw .and. stride >= kh .and. pad == 0) then
            max_err = maxval(abs(output - input))
            if (max_err > 1e-5) then
                print *, "  FAIL config: roundtrip error =", max_err
                return
            end if
        end if

        deallocate(input, col, output)
        ok = .true.
    end function

end program
