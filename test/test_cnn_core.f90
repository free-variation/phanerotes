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

    ! conv_forward tests
    call test_conv_forward_dimensions()
    call test_conv_forward_zero_input()
    call test_conv_forward_known_values()
    call test_conv_forward_naive_equivalence()
    call test_conv_forward_linearity()

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

    ! ============ conv_forward tests ============

    subroutine init_layer(layer, in_c, out_c, kw, kh, stride, pad)
        type(conv_layer), intent(out) :: layer
        integer, intent(in) :: in_c, out_c, kw, kh, stride, pad

        layer%in_channels = in_c
        layer%out_channels = out_c
        layer%kernel_width = kw
        layer%kernel_height = kh
        layer%stride = stride
        layer%padding = pad

        allocate(layer%weights(out_c, in_c, kw, kh))
        allocate(layer%bias(out_c))
        allocate(layer%weights_grad(out_c, in_c, kw, kh))
        allocate(layer%bias_grad(out_c))

        layer%weights = 0.0
        layer%bias = 0.0
        layer%weights_grad = 0.0
        layer%bias_grad = 0.0
    end subroutine

    subroutine test_conv_forward_dimensions()
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:), output(:,:,:)
        integer :: out_w, out_h

        ! 3 input channels, 8 output channels, 3x3 kernel, stride 1, pad 1
        call init_layer(layer, 3, 8, 3, 3, 1, 1)
        call random_number(layer%weights)
        call random_number(layer%bias)

        ! 16x16 input
        allocate(input(3, 16, 16))
        call random_number(input)

        call conv_forward(layer, input, output)

        out_w = (16 + 2*1 - 3) / 1 + 1  ! = 16
        out_h = (16 + 2*1 - 3) / 1 + 1  ! = 16

        if (size(output, 1) /= 8 .or. size(output, 2) /= 16 .or. size(output, 3) /= 16) then
            print *, "FAIL conv_forward_dimensions: expected (8,16,16), got", &
                     size(output,1), size(output,2), size(output,3)
            error stop
        end if

        print *, "PASS: conv_forward dimensions"
    end subroutine

    subroutine test_conv_forward_zero_input()
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:), output(:,:,:)
        integer :: oc, oi, oj
        real :: max_err

        call init_layer(layer, 2, 4, 3, 3, 1, 1)
        call random_number(layer%weights)
        layer%bias = [1.0, 2.0, 3.0, 4.0]

        allocate(input(2, 8, 8))
        input = 0.0

        call conv_forward(layer, input, output)

        ! Output should equal bias at every spatial position
        max_err = 0.0
        do oc = 1, 4
            do oj = 1, size(output, 3)
                do oi = 1, size(output, 2)
                    max_err = max(max_err, abs(output(oc, oi, oj) - layer%bias(oc)))
                end do
            end do
        end do

        if (max_err > 1e-5) then
            print *, "FAIL conv_forward_zero_input: max error =", max_err
            error stop
        end if

        print *, "PASS: conv_forward zero input"
    end subroutine

    subroutine test_conv_forward_known_values()
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:), output(:,:,:)
        real :: expected

        ! Simplest case: 1 in, 1 out, 2x2 kernel, stride 1, no padding
        ! Input: 3x3, all ones
        ! Weights: 2x2, all ones -> each output = 4 * 1 = 4
        ! Bias: 0.5 -> each output = 4.5

        call init_layer(layer, 1, 1, 2, 2, 1, 0)
        layer%weights = 1.0
        layer%bias = [0.5]

        allocate(input(1, 3, 3))
        input = 1.0

        call conv_forward(layer, input, output)

        ! Output should be 2x2, all 4.5
        if (size(output, 2) /= 2 .or. size(output, 3) /= 2) then
            print *, "FAIL conv_forward_known_values: wrong output size"
            error stop
        end if

        expected = 4.0 + 0.5
        if (abs(output(1,1,1) - expected) > 1e-5 .or. &
            abs(output(1,2,1) - expected) > 1e-5 .or. &
            abs(output(1,1,2) - expected) > 1e-5 .or. &
            abs(output(1,2,2) - expected) > 1e-5) then
            print *, "FAIL conv_forward_known_values: expected", expected, "got", output
            error stop
        end if

        print *, "PASS: conv_forward known values"
    end subroutine

    subroutine test_conv_forward_naive_equivalence()
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:), output(:,:,:)
        real, allocatable :: expected(:,:,:), padded(:,:,:)
        integer :: in_c, out_c, w, h, kw, kh, stride, pad
        integer :: out_w, out_h, oc, ic, oi, oj, ki, kj
        real :: sum_val, max_err

        in_c = 3
        out_c = 4
        w = 8
        h = 8
        kw = 3
        kh = 3
        stride = 1
        pad = 1

        call init_layer(layer, in_c, out_c, kw, kh, stride, pad)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input(in_c, w, h))
        call random_number(input)

        ! Run conv_forward
        call conv_forward(layer, input, output)

        ! Naive implementation
        out_w = (w + 2*pad - kw) / stride + 1
        out_h = (h + 2*pad - kh) / stride + 1

        allocate(padded(in_c, w + 2*pad, h + 2*pad))
        padded = 0.0
        padded(:, pad+1:pad+w, pad+1:pad+h) = input

        allocate(expected(out_c, out_w, out_h))

        do oc = 1, out_c
            do oj = 1, out_h
                do oi = 1, out_w
                    sum_val = layer%bias(oc)
                    do ic = 1, in_c
                        do kj = 1, kh
                            do ki = 1, kw
                                sum_val = sum_val + layer%weights(oc, ic, ki, kj) * &
                                          padded(ic, (oi-1)*stride + ki, (oj-1)*stride + kj)
                            end do
                        end do
                    end do
                    expected(oc, oi, oj) = sum_val
                end do
            end do
        end do

        max_err = maxval(abs(output - expected))

        if (max_err > 1e-4) then
            print *, "FAIL conv_forward_naive_equivalence: max error =", max_err
            error stop
        end if

        print *, "PASS: conv_forward naive equivalence"
    end subroutine

    subroutine test_conv_forward_linearity()
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:), output1(:,:,:), output2(:,:,:)
        real :: alpha, max_err
        integer :: oc, oi, oj

        call init_layer(layer, 2, 3, 3, 3, 1, 1)
        call random_number(layer%weights)
        layer%bias = 0.0  ! Zero bias for pure linearity test

        allocate(input(2, 8, 8))
        call random_number(input)

        alpha = 2.5

        ! conv(input)
        call conv_forward(layer, input, output1)

        ! conv(alpha * input)
        call conv_forward(layer, alpha * input, output2)

        ! Should have: output2 = alpha * output1
        max_err = maxval(abs(output2 - alpha * output1))

        if (max_err > 1e-4) then
            print *, "FAIL conv_forward_linearity: max error =", max_err
            error stop
        end if

        print *, "PASS: conv_forward linearity"
    end subroutine

end program
