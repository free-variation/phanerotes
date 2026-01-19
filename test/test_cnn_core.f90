program test_cnn_core
    use cnn_core
    use nn
    use cnn_autoencoder
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

    ! conv_backward tests
    call test_conv_backward_dimensions()
    call test_conv_backward_zero_grad()
    call test_conv_backward_weights_numerical()
    call test_conv_backward_bias_numerical()
    call test_conv_backward_input_numerical()

    ! activation tests
    call test_relu_forward()
    call test_relu_backward()
    call test_sigmoid_forward()
    call test_sigmoid_backward_numerical()

    ! upsample tests
    call test_upsample_dimensions()
    call test_upsample_values()
    call test_upsample_backward_dimensions()
    call test_upsample_backward_values()
    call test_upsample_roundtrip()

    ! autoencoder init tests
    call test_autoencoder_init_structure()
    call test_autoencoder_init_channels()
    call test_autoencoder_init_weights()
    call test_autoencoder_init_strides()

    ! autoencoder forward tests
    call test_autoencoder_forward_dimensions()
    call test_autoencoder_forward_output_range()

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

    subroutine test_init_layer(layer, in_c, out_c, kw, kh, stride, pad)
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
        call test_init_layer(layer, 3, 8, 3, 3, 1, 1)
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

        call test_init_layer(layer, 2, 4, 3, 3, 1, 1)
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

        call test_init_layer(layer, 1, 1, 2, 2, 1, 0)
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

        call test_init_layer(layer, in_c, out_c, kw, kh, stride, pad)
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

        call test_init_layer(layer, 2, 3, 3, 3, 1, 1)
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

    ! ============ conv_backward tests ============

    subroutine test_conv_backward_dimensions()
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:), output(:,:,:)
        real, allocatable :: grad_output(:,:,:), grad_input(:,:,:)

        call test_init_layer(layer, 3, 8, 3, 3, 1, 1)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input(3, 16, 16))
        call random_number(input)

        call conv_forward(layer, input, output)

        allocate(grad_output(8, 16, 16))
        call random_number(grad_output)

        call conv_backward(layer, grad_output, grad_input)

        ! Check grad_input dimensions match input
        if (size(grad_input, 1) /= 3 .or. size(grad_input, 2) /= 16 .or. size(grad_input, 3) /= 16) then
            print *, "FAIL conv_backward_dimensions: grad_input shape wrong"
            error stop
        end if

        ! Check weights_grad dimensions match weights
        if (size(layer%weights_grad, 1) /= 8 .or. size(layer%weights_grad, 2) /= 3 .or. &
            size(layer%weights_grad, 3) /= 3 .or. size(layer%weights_grad, 4) /= 3) then
            print *, "FAIL conv_backward_dimensions: weights_grad shape wrong"
            error stop
        end if

        ! Check bias_grad dimensions
        if (size(layer%bias_grad) /= 8) then
            print *, "FAIL conv_backward_dimensions: bias_grad shape wrong"
            error stop
        end if

        print *, "PASS: conv_backward dimensions"
    end subroutine

    subroutine test_conv_backward_zero_grad()
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:), output(:,:,:)
        real, allocatable :: grad_output(:,:,:), grad_input(:,:,:)

        call test_init_layer(layer, 2, 4, 3, 3, 1, 1)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input(2, 8, 8))
        call random_number(input)

        call conv_forward(layer, input, output)

        allocate(grad_output(4, 8, 8))
        grad_output = 0.0

        call conv_backward(layer, grad_output, grad_input)

        if (maxval(abs(grad_input)) > 1e-6) then
            print *, "FAIL conv_backward_zero_grad: grad_input not zero"
            error stop
        end if

        if (maxval(abs(layer%weights_grad)) > 1e-6) then
            print *, "FAIL conv_backward_zero_grad: weights_grad not zero"
            error stop
        end if

        if (maxval(abs(layer%bias_grad)) > 1e-6) then
            print *, "FAIL conv_backward_zero_grad: bias_grad not zero"
            error stop
        end if

        print *, "PASS: conv_backward zero grad"
    end subroutine

    subroutine test_conv_backward_weights_numerical()
        ! Numerical gradient check: perturb each weight, measure change in loss
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:), output(:,:,:), output_plus(:,:,:), output_minus(:,:,:)
        real, allocatable :: grad_output(:,:,:), grad_input(:,:,:)
        real :: eps, numerical_grad, analytical_grad, max_err, loss_plus, loss_minus
        real :: original_weight
        integer :: oc, ic, ki, kj

        eps = 1e-3
        max_err = 0.0

        call test_init_layer(layer, 2, 3, 2, 2, 1, 0)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input(2, 4, 4))
        call random_number(input)

        ! Forward pass to get output shape
        call conv_forward(layer, input, output)

        ! Use ones as grad_output for simpler gradient computation
        allocate(grad_output(size(output,1), size(output,2), size(output,3)))
        grad_output = 1.0

        call conv_backward(layer, grad_output, grad_input)

        ! Check a few weight gradients numerically
        do oc = 1, 2
            do ic = 1, 2
                do ki = 1, 2
                    do kj = 1, 2
                        original_weight = layer%weights(oc, ic, ki, kj)

                        ! f(w + eps)
                        layer%weights(oc, ic, ki, kj) = original_weight + eps
                        call conv_forward(layer, input, output_plus)
                        loss_plus = sum(output_plus)

                        ! f(w - eps)
                        layer%weights(oc, ic, ki, kj) = original_weight - eps
                        call conv_forward(layer, input, output_minus)
                        loss_minus = sum(output_minus)

                        ! Restore weight
                        layer%weights(oc, ic, ki, kj) = original_weight

                        numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)
                        analytical_grad = layer%weights_grad(oc, ic, ki, kj)

                        if (abs(numerical_grad - analytical_grad) > max_err) then
                            max_err = abs(numerical_grad - analytical_grad)
                        end if
                    end do
                end do
            end do
        end do

        if (max_err > 1e-2) then
            print *, "FAIL conv_backward_weights_numerical: max error =", max_err
            error stop
        end if

        print *, "PASS: conv_backward weights numerical (max err:", max_err, ")"
    end subroutine

    subroutine test_conv_backward_bias_numerical()
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:), output(:,:,:), output_plus(:,:,:), output_minus(:,:,:)
        real, allocatable :: grad_output(:,:,:), grad_input(:,:,:)
        real :: eps, numerical_grad, analytical_grad, max_err, loss_plus, loss_minus
        real :: original_bias
        integer :: oc

        eps = 1e-3
        max_err = 0.0

        call test_init_layer(layer, 2, 3, 2, 2, 1, 0)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input(2, 4, 4))
        call random_number(input)

        call conv_forward(layer, input, output)

        allocate(grad_output(size(output,1), size(output,2), size(output,3)))
        grad_output = 1.0

        call conv_backward(layer, grad_output, grad_input)

        do oc = 1, 3
            original_bias = layer%bias(oc)

            layer%bias(oc) = original_bias + eps
            call conv_forward(layer, input, output_plus)
            loss_plus = sum(output_plus)

            layer%bias(oc) = original_bias - eps
            call conv_forward(layer, input, output_minus)
            loss_minus = sum(output_minus)

            layer%bias(oc) = original_bias

            numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)
            analytical_grad = layer%bias_grad(oc)

            max_err = max(max_err, abs(numerical_grad - analytical_grad))
        end do

        if (max_err > 1e-2) then
            print *, "FAIL conv_backward_bias_numerical: max error =", max_err
            error stop
        end if

        print *, "PASS: conv_backward bias numerical (max err:", max_err, ")"
    end subroutine

    subroutine test_conv_backward_input_numerical()
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:), output(:,:,:), output_plus(:,:,:), output_minus(:,:,:)
        real, allocatable :: grad_output(:,:,:), grad_input(:,:,:)
        real :: eps, numerical_grad, analytical_grad, max_err, loss_plus, loss_minus
        real :: original_input
        integer :: ic, ii, ij

        eps = 1e-3
        max_err = 0.0

        call test_init_layer(layer, 2, 3, 2, 2, 1, 0)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input(2, 4, 4))
        call random_number(input)

        call conv_forward(layer, input, output)

        allocate(grad_output(size(output,1), size(output,2), size(output,3)))
        grad_output = 1.0

        call conv_backward(layer, grad_output, grad_input)

        ! Check a subset of input gradients
        do ic = 1, 2
            do ii = 1, 3
                do ij = 1, 3
                    original_input = input(ic, ii, ij)

                    input(ic, ii, ij) = original_input + eps
                    call conv_forward(layer, input, output_plus)
                    loss_plus = sum(output_plus)

                    input(ic, ii, ij) = original_input - eps
                    call conv_forward(layer, input, output_minus)
                    loss_minus = sum(output_minus)

                    input(ic, ii, ij) = original_input

                    numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)
                    analytical_grad = grad_input(ic, ii, ij)

                    max_err = max(max_err, abs(numerical_grad - analytical_grad))
                end do
            end do
        end do

        if (max_err > 1e-2) then
            print *, "FAIL conv_backward_input_numerical: max error =", max_err
            error stop
        end if

        print *, "PASS: conv_backward input numerical (max err:", max_err, ")"
    end subroutine

    ! ============ activation tests ============

    subroutine test_relu_forward()
        real :: input(2, 3, 3), output(2, 3, 3)

        input(1, :, :) = reshape([-1.0, 0.0, 1.0, &
                                  -2.0, 0.5, 2.0, &
                                  -0.1, 0.0, 0.1], [3, 3])
        input(2, :, :) = reshape([1.0, -1.0, 0.0, &
                                  3.0, -3.0, 0.0, &
                                  0.5, -0.5, 0.0], [3, 3])

        output = relu_forward(input)

        ! Check negatives become zero
        if (output(1, 1, 1) /= 0.0 .or. output(1, 1, 2) /= 0.0 .or. output(1, 1, 3) /= 0.0) then
            print *, "FAIL relu_forward: negatives not zeroed"
            error stop
        end if

        ! Check positives pass through
        if (abs(output(1, 3, 1) - 1.0) > 1e-6 .or. abs(output(1, 3, 2) - 2.0) > 1e-6) then
            print *, "FAIL relu_forward: positives not passed"
            error stop
        end if

        ! Check zeros stay zero
        if (output(1, 2, 1) /= 0.0 .or. output(2, 3, 1) /= 0.0) then
            print *, "FAIL relu_forward: zeros changed"
            error stop
        end if

        print *, "PASS: relu_forward"
    end subroutine

    subroutine test_relu_backward()
        real :: x(1, 4, 1), grad_out(1, 4, 1), grad_in(1, 4, 1)

        x(1, :, 1) = [-1.0, 0.0, 1.0, 2.0]
        grad_out(1, :, 1) = [5.0, 5.0, 5.0, 5.0]

        grad_in = relu_backward(x, grad_out)

        ! Negative x -> gradient blocked
        if (grad_in(1, 1, 1) /= 0.0) then
            print *, "FAIL relu_backward: gradient not blocked for negative"
            error stop
        end if

        ! Zero x -> gradient blocked (x > 0 is false)
        if (grad_in(1, 2, 1) /= 0.0) then
            print *, "FAIL relu_backward: gradient not blocked for zero"
            error stop
        end if

        ! Positive x -> gradient passes
        if (abs(grad_in(1, 3, 1) - 5.0) > 1e-6 .or. abs(grad_in(1, 4, 1) - 5.0) > 1e-6) then
            print *, "FAIL relu_backward: gradient not passed for positive"
            error stop
        end if

        print *, "PASS: relu_backward"
    end subroutine

    subroutine test_sigmoid_forward()
        real :: input(1, 3, 1), output(1, 3, 1)

        input(1, :, 1) = [0.0, 10.0, -10.0]

        output = sigmoid_forward(input)

        ! sigmoid(0) = 0.5
        if (abs(output(1, 1, 1) - 0.5) > 1e-6) then
            print *, "FAIL sigmoid_forward: sigmoid(0) /= 0.5, got", output(1, 1, 1)
            error stop
        end if

        ! sigmoid(10) ~ 1
        if (abs(output(1, 2, 1) - 1.0) > 1e-4) then
            print *, "FAIL sigmoid_forward: sigmoid(10) not near 1, got", output(1, 2, 1)
            error stop
        end if

        ! sigmoid(-10) ~ 0
        if (abs(output(1, 3, 1)) > 1e-4) then
            print *, "FAIL sigmoid_forward: sigmoid(-10) not near 0, got", output(1, 3, 1)
            error stop
        end if

        print *, "PASS: sigmoid_forward"
    end subroutine

    subroutine test_sigmoid_backward_numerical()
        real, allocatable :: x(:,:,:), y(:,:,:), grad_out(:,:,:), grad_in(:,:,:)
        real :: eps, numerical_grad, analytical_grad, max_err
        real :: y_plus, y_minus, loss_plus, loss_minus
        integer :: i, j

        eps = 1e-4
        max_err = 0.0

        allocate(x(1, 4, 4))
        call random_number(x)
        x = x * 4.0 - 2.0  ! range [-2, 2]

        y = sigmoid_forward(x)

        allocate(grad_out(1, 4, 4))
        grad_out = 1.0

        grad_in = sigmoid_backward(y, grad_out)

        ! Numerical check
        do j = 1, 4
            do i = 1, 4
                y_plus = 1.0 / (1.0 + exp(-(x(1, i, j) + eps)))
                y_minus = 1.0 / (1.0 + exp(-(x(1, i, j) - eps)))
                loss_plus = y_plus
                loss_minus = y_minus

                numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)
                analytical_grad = grad_in(1, i, j)

                max_err = max(max_err, abs(numerical_grad - analytical_grad))
            end do
        end do

        if (max_err > 1e-3) then
            print *, "FAIL sigmoid_backward_numerical: max error =", max_err
            error stop
        end if

        print *, "PASS: sigmoid_backward numerical (max err:", max_err, ")"
    end subroutine

    ! ============ upsample tests ============

    subroutine test_upsample_dimensions()
        real, allocatable :: input(:,:,:), output(:,:,:)

        allocate(input(3, 4, 5))
        call random_number(input)

        output = upsample(input, 2)

        if (size(output, 1) /= 3 .or. size(output, 2) /= 8 .or. size(output, 3) /= 10) then
            print *, "FAIL upsample_dimensions: expected (3,8,10), got", &
                     size(output,1), size(output,2), size(output,3)
            error stop
        end if

        print *, "PASS: upsample dimensions"
    end subroutine

    subroutine test_upsample_values()
        real :: input(1, 2, 2), output(1, 4, 4)

        input(1, :, :) = reshape([1.0, 2.0, 3.0, 4.0], [2, 2])

        output = upsample(input, 2)

        ! Each input value should be replicated in a 2x2 block
        ! input(1,1) = 1.0 -> output(1:2, 1:2) = 1.0
        if (output(1,1,1) /= 1.0 .or. output(1,2,1) /= 1.0 .or. &
            output(1,1,2) /= 1.0 .or. output(1,2,2) /= 1.0) then
            print *, "FAIL upsample_values: top-left block wrong"
            error stop
        end if

        ! input(2,1) = 2.0 -> output(3:4, 1:2) = 2.0
        if (output(1,3,1) /= 2.0 .or. output(1,4,1) /= 2.0 .or. &
            output(1,3,2) /= 2.0 .or. output(1,4,2) /= 2.0) then
            print *, "FAIL upsample_values: top-right block wrong"
            error stop
        end if

        ! input(1,2) = 3.0 -> output(1:2, 3:4) = 3.0
        if (output(1,1,3) /= 3.0 .or. output(1,2,3) /= 3.0 .or. &
            output(1,1,4) /= 3.0 .or. output(1,2,4) /= 3.0) then
            print *, "FAIL upsample_values: bottom-left block wrong"
            error stop
        end if

        ! input(2,2) = 4.0 -> output(3:4, 3:4) = 4.0
        if (output(1,3,3) /= 4.0 .or. output(1,4,3) /= 4.0 .or. &
            output(1,3,4) /= 4.0 .or. output(1,4,4) /= 4.0) then
            print *, "FAIL upsample_values: bottom-right block wrong"
            error stop
        end if

        print *, "PASS: upsample values"
    end subroutine

    subroutine test_upsample_backward_dimensions()
        real, allocatable :: grad_output(:,:,:), grad_input(:,:,:)

        allocate(grad_output(3, 8, 10))
        call random_number(grad_output)

        grad_input = upsample_backward(grad_output, 2)

        if (size(grad_input, 1) /= 3 .or. size(grad_input, 2) /= 4 .or. size(grad_input, 3) /= 5) then
            print *, "FAIL upsample_backward_dimensions: expected (3,4,5), got", &
                     size(grad_input,1), size(grad_input,2), size(grad_input,3)
            error stop
        end if

        print *, "PASS: upsample_backward dimensions"
    end subroutine

    subroutine test_upsample_backward_values()
        real :: grad_output(1, 4, 4), grad_input(1, 2, 2)

        ! Set each 2x2 block to different values that sum to known results
        grad_output(1, 1:2, 1:2) = 1.0  ! sum = 4
        grad_output(1, 3:4, 1:2) = 2.0  ! sum = 8
        grad_output(1, 1:2, 3:4) = 0.5  ! sum = 2
        grad_output(1, 3:4, 3:4) = 0.25 ! sum = 1

        grad_input = upsample_backward(grad_output, 2)

        if (abs(grad_input(1,1,1) - 4.0) > 1e-6) then
            print *, "FAIL upsample_backward_values: (1,1) expected 4.0, got", grad_input(1,1,1)
            error stop
        end if

        if (abs(grad_input(1,2,1) - 8.0) > 1e-6) then
            print *, "FAIL upsample_backward_values: (2,1) expected 8.0, got", grad_input(1,2,1)
            error stop
        end if

        if (abs(grad_input(1,1,2) - 2.0) > 1e-6) then
            print *, "FAIL upsample_backward_values: (1,2) expected 2.0, got", grad_input(1,1,2)
            error stop
        end if

        if (abs(grad_input(1,2,2) - 1.0) > 1e-6) then
            print *, "FAIL upsample_backward_values: (2,2) expected 1.0, got", grad_input(1,2,2)
            error stop
        end if

        print *, "PASS: upsample_backward values"
    end subroutine

    subroutine test_upsample_roundtrip()
        ! If we upsample then backward with grad=1, each input contributes to factor^2 outputs
        ! So backward(ones, factor) should give factor^2 at each position
        real, allocatable :: input(:,:,:), upsampled(:,:,:), grad(:,:,:)
        integer :: factor

        factor = 3

        allocate(input(2, 4, 5))
        call random_number(input)

        upsampled = upsample(input, factor)

        allocate(grad(2, 4*factor, 5*factor))
        grad = 1.0

        ! backward of all-ones gives factor^2 at each input position
        ! This verifies the gradient correctly accumulates from all factor^2 outputs
        grad = upsample_backward(grad, factor)

        if (abs(grad(1,1,1) - real(factor*factor)) > 1e-6) then
            print *, "FAIL upsample_roundtrip: expected", factor*factor, "got", grad(1,1,1)
            error stop
        end if

        if (maxval(abs(grad - real(factor*factor))) > 1e-6) then
            print *, "FAIL upsample_roundtrip: not all values are factor^2"
            error stop
        end if

        print *, "PASS: upsample roundtrip"
    end subroutine

    ! ============ autoencoder init tests ============

    subroutine test_autoencoder_init_structure()
        type(autoencoder_config) :: config
        type(autoencoder) :: net

        config%input_channels = 3
        config%num_layers = 4
        config%base_channels = 64
        config%max_channels = 512
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        if (size(net%encoder) /= 4) then
            print *, "FAIL autoencoder_init_structure: encoder size wrong"
            error stop
        end if

        if (size(net%decoder) /= 4) then
            print *, "FAIL autoencoder_init_structure: decoder size wrong"
            error stop
        end if

        print *, "PASS: autoencoder_init structure"
    end subroutine

    subroutine test_autoencoder_init_channels()
        type(autoencoder_config) :: config
        type(autoencoder) :: net

        config%input_channels = 3
        config%num_layers = 4
        config%base_channels = 64
        config%max_channels = 512
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        ! Encoder: 3->128, 128->256, 256->512, 512->512
        if (net%encoder(1)%in_channels /= 3 .or. net%encoder(1)%out_channels /= 128) then
            print *, "FAIL autoencoder_init_channels: encoder(1) wrong", &
                     net%encoder(1)%in_channels, net%encoder(1)%out_channels
            error stop
        end if

        if (net%encoder(2)%in_channels /= 128 .or. net%encoder(2)%out_channels /= 256) then
            print *, "FAIL autoencoder_init_channels: encoder(2) wrong"
            error stop
        end if

        if (net%encoder(3)%in_channels /= 256 .or. net%encoder(3)%out_channels /= 512) then
            print *, "FAIL autoencoder_init_channels: encoder(3) wrong"
            error stop
        end if

        if (net%encoder(4)%in_channels /= 512 .or. net%encoder(4)%out_channels /= 512) then
            print *, "FAIL autoencoder_init_channels: encoder(4) wrong"
            error stop
        end if

        ! Decoder mirrors encoder: 512->512, 512->256, 256->128, 128->3
        if (net%decoder(1)%in_channels /= 512 .or. net%decoder(1)%out_channels /= 512) then
            print *, "FAIL autoencoder_init_channels: decoder(1) wrong"
            error stop
        end if

        if (net%decoder(2)%in_channels /= 512 .or. net%decoder(2)%out_channels /= 256) then
            print *, "FAIL autoencoder_init_channels: decoder(2) wrong"
            error stop
        end if

        if (net%decoder(3)%in_channels /= 256 .or. net%decoder(3)%out_channels /= 128) then
            print *, "FAIL autoencoder_init_channels: decoder(3) wrong"
            error stop
        end if

        if (net%decoder(4)%in_channels /= 128 .or. net%decoder(4)%out_channels /= 3) then
            print *, "FAIL autoencoder_init_channels: decoder(4) wrong"
            error stop
        end if

        print *, "PASS: autoencoder_init channels"
    end subroutine

    subroutine test_autoencoder_init_weights()
        type(autoencoder_config) :: config
        type(autoencoder) :: net
        integer :: i
        real :: weight_sum

        config%input_channels = 3
        config%num_layers = 3
        config%base_channels = 32
        config%max_channels = 256
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        do i = 1, 3
            ! Check weights allocated with correct shape
            if (size(net%encoder(i)%weights, 1) /= net%encoder(i)%out_channels .or. &
                size(net%encoder(i)%weights, 2) /= net%encoder(i)%in_channels .or. &
                size(net%encoder(i)%weights, 3) /= 3 .or. &
                size(net%encoder(i)%weights, 4) /= 3) then
                print *, "FAIL autoencoder_init_weights: encoder weights shape wrong at layer", i
                error stop
            end if

            ! Check bias allocated and zeroed
            if (size(net%encoder(i)%bias) /= net%encoder(i)%out_channels) then
                print *, "FAIL autoencoder_init_weights: encoder bias size wrong at layer", i
                error stop
            end if

            if (maxval(abs(net%encoder(i)%bias)) > 1e-6) then
                print *, "FAIL autoencoder_init_weights: encoder bias not zero at layer", i
                error stop
            end if

            ! Check weights are initialized (not all zero)
            weight_sum = sum(abs(net%encoder(i)%weights))
            if (weight_sum < 1e-6) then
                print *, "FAIL autoencoder_init_weights: encoder weights all zero at layer", i
                error stop
            end if
        end do

        print *, "PASS: autoencoder_init weights"
    end subroutine

    subroutine test_autoencoder_init_strides()
        type(autoencoder_config) :: config
        type(autoencoder) :: net
        integer :: i

        config%input_channels = 3
        config%num_layers = 3
        config%base_channels = 32
        config%max_channels = 256
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        ! Encoder should have stride = config%stride
        do i = 1, 3
            if (net%encoder(i)%stride /= 2) then
                print *, "FAIL autoencoder_init_strides: encoder stride wrong at layer", i
                error stop
            end if
        end do

        ! Decoder should have stride = 1
        do i = 1, 3
            if (net%decoder(i)%stride /= 1) then
                print *, "FAIL autoencoder_init_strides: decoder stride wrong at layer", i, &
                         "got", net%decoder(i)%stride
                error stop
            end if
        end do

        print *, "PASS: autoencoder_init strides"
    end subroutine

    subroutine test_autoencoder_forward_dimensions()
        type(autoencoder_config) :: config
        type(autoencoder) :: net
        real, allocatable :: input(:,:,:), latent(:,:,:), output(:,:,:)

        config%input_channels = 3
        config%num_layers = 3
        config%base_channels = 32
        config%max_channels = 256
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        ! Input: 32x32x3, after 3 stride-2 layers: 4x4
        allocate(input(3, 32, 32))
        call random_number(input)

        call autoencoder_forward(net, input, latent, output)

        ! Latent should be (max_channels, 4, 4) after 3 stride-2 downsamples
        if (size(latent, 1) /= 256 .or. size(latent, 2) /= 4 .or. size(latent, 3) /= 4) then
            print *, "FAIL autoencoder_forward_dimensions: latent shape wrong", &
                     size(latent, 1), size(latent, 2), size(latent, 3)
            error stop
        end if

        ! Output should match input dimensions
        if (size(output, 1) /= 3 .or. size(output, 2) /= 32 .or. size(output, 3) /= 32) then
            print *, "FAIL autoencoder_forward_dimensions: output shape wrong", &
                     size(output, 1), size(output, 2), size(output, 3)
            error stop
        end if

        print *, "PASS: autoencoder_forward dimensions"
    end subroutine

    subroutine test_autoencoder_forward_output_range()
        type(autoencoder_config) :: config
        type(autoencoder) :: net
        real, allocatable :: input(:,:,:), latent(:,:,:), output(:,:,:)
        real :: min_val, max_val

        config%input_channels = 1
        config%num_layers = 2
        config%base_channels = 16
        config%max_channels = 64
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        allocate(input(1, 16, 16))
        call random_number(input)

        call autoencoder_forward(net, input, latent, output)

        ! Output should be in [0, 1] due to sigmoid
        min_val = minval(output)
        max_val = maxval(output)

        if (min_val < 0.0 .or. max_val > 1.0) then
            print *, "FAIL autoencoder_forward_output_range: output not in [0,1]", min_val, max_val
            error stop
        end if

        print *, "PASS: autoencoder_forward output range"
    end subroutine

end program
