program test_cnn_core
    use cnn_core
    use nn
    use cnn_autoencoder
    use train
    implicit none

    ! im2col tests
    call test_im2col_basic()
    call test_im2col_padding()
    call test_im2col_batched()

    ! col2im tests
    call test_col2im_roundtrip()
    call test_col2im_accumulation()
    call test_col2im_batched()

    ! Combined tests
    call test_full_configurations()

    ! conv_forward tests
    call test_conv_forward_dimensions()
    call test_conv_forward_zero_input()
    call test_conv_forward_known_values()
    call test_conv_forward_naive_equivalence()
    call test_conv_forward_linearity()
    call test_conv_forward_batched()

    ! conv_backward tests
    call test_conv_backward_dimensions()
    call test_conv_backward_zero_grad()
    call test_conv_backward_weights_numerical()
    call test_conv_backward_bias_numerical()
    call test_conv_backward_input_numerical()
    call test_conv_backward_batched()

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
    call test_autoencoder_forward_batched()

    ! autoencoder backward tests
    call test_autoencoder_backward_numerical()
    call test_autoencoder_backward_batched()

    ! loss function tests
    call test_mse_loss()
    call test_mse_loss_grad()

    ! sgd tests
    call test_sgd_update()

    ! weights io tests
    call test_save_load_weights()

    ! training test
    call test_training_loss_decreases()
    call test_training_batched()

contains

    subroutine test_im2col_basic()
        real, allocatable :: input(:,:,:,:)
        real, allocatable :: result(:,:)
        real :: expected(4, 4)
        integer :: i, j, errors

        ! batch=1, 1 channel, 4x4 input, 2x2 kernel, stride 2, no padding
        allocate(input(1, 1, 4, 4))
        do j = 1, 4
            do i = 1, 4
                input(1, 1, i, j) = real((j - 1) * 4 + i)
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
        real, allocatable :: input(:,:,:,:)
        real, allocatable :: result(:,:)

        allocate(input(1, 1, 3, 3))
        input(1, 1, :, :) = reshape([1.0, 2.0, 3.0, &
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

    subroutine test_im2col_batched()
        real, allocatable :: input(:,:,:,:)
        real, allocatable :: result(:,:)
        integer :: batch_size, out_w, out_h

        ! batch=3, 2 channels, 4x4 input, 2x2 kernel, stride 2, no padding
        batch_size = 3
        allocate(input(batch_size, 2, 4, 4))
        call random_number(input)

        result = im2col(input, 2, 2, 2, 0)

        out_w = 2
        out_h = 2

        ! Output should be (C*kw*kh, out_w*out_h*batch) = (8, 12)
        if (size(result, 1) /= 8 .or. size(result, 2) /= out_w * out_h * batch_size) then
            print *, "FAIL im2col_batched: wrong dimensions, expected (8, 12), got", &
                     size(result, 1), size(result, 2)
            error stop
        end if

        print *, "PASS: im2col batched"
        deallocate(input, result)
    end subroutine

    subroutine test_col2im_batched()
        real, allocatable :: input(:,:,:,:)
        real, allocatable :: col(:,:)
        real, allocatable :: reconstructed(:,:,:,:)
        integer :: batch_size, b
        real :: max_err

        ! batch=3, 1 channel, 4x4 input, 2x2 kernel, stride 2 (non-overlapping)
        batch_size = 3
        allocate(input(batch_size, 1, 4, 4))
        call random_number(input)

        col = im2col(input, 2, 2, 2, 0)
        reconstructed = col2im(col, batch_size, 1, 4, 4, 2, 2, 2, 0)

        ! Check dimensions
        if (size(reconstructed, 1) /= batch_size .or. size(reconstructed, 2) /= 1 .or. &
            size(reconstructed, 3) /= 4 .or. size(reconstructed, 4) /= 4) then
            print *, "FAIL col2im_batched: wrong dimensions"
            error stop
        end if

        ! Check roundtrip (non-overlapping patches should reconstruct exactly)
        max_err = maxval(abs(reconstructed - input))
        if (max_err > 1e-5) then
            print *, "FAIL col2im_batched: roundtrip error =", max_err
            error stop
        end if

        print *, "PASS: col2im batched"
        deallocate(input, col, reconstructed)
    end subroutine

    subroutine test_col2im_roundtrip()
        real, allocatable :: input(:,:,:,:)
        real, allocatable :: col_matrix(:,:)
        real, allocatable :: reconstructed(:,:,:,:)
        integer :: i, j, errors

        allocate(input(1, 1, 4, 4))
        do j = 1, 4
            do i = 1, 4
                input(1, 1, i, j) = real((j - 1) * 4 + i)
            end do
        end do

        col_matrix = im2col(input, 2, 2, 2, 0)
        reconstructed = col2im(col_matrix, 1, 1, 4, 4, 2, 2, 2, 0)

        errors = 0
        do j = 1, 4
            do i = 1, 4
                if (abs(reconstructed(1, 1, i, j) - input(1, 1, i, j)) > 1e-6) then
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
        real, allocatable :: result(:,:,:,:)

        ! batch=1, 3x3 output, 2x2 kernel, stride 1 -> 4 overlapping patches
        allocate(col_matrix(4, 4))
        col_matrix = 1.0

        result = col2im(col_matrix, 1, 1, 3, 3, 2, 2, 1, 0)

        ! Center (2,2) should accumulate 4x, edges 2x, corners 1x
        if (abs(result(1, 1, 1, 1) - 1.0) > 1e-6 .or. &
            abs(result(1, 1, 2, 1) - 2.0) > 1e-6 .or. &
            abs(result(1, 1, 2, 2) - 4.0) > 1e-6 .or. &
            abs(result(1, 1, 3, 3) - 1.0) > 1e-6) then
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

        real, allocatable :: input(:,:,:,:), col(:,:), output(:,:,:,:)
        integer :: i, j, c, out_w, out_h
        real :: max_err

        ok = .false.

        allocate(input(1, nc, w, h))
        do c = 1, nc
            do j = 1, h
                do i = 1, w
                    input(1, c, i, j) = real(c * 1000 + j * 100 + i)
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

        output = col2im(col, 1, nc, w, h, kw, kh, stride, pad)

        if (size(output, 1) /= 1 .or. size(output, 2) /= nc .or. &
            size(output, 3) /= w .or. size(output, 4) /= h) then
            print *, "  FAIL config: col2im dims"
            return
        end if

        if (stride >= kw .and. stride >= kh .and. pad == 0) then
            max_err = maxval(abs(output(1,:,:,:) - input(1,:,:,:)))
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
        real, allocatable :: input(:,:,:,:), output(:,:,:,:)
        integer :: out_w, out_h

        ! 3 input channels, 8 output channels, 3x3 kernel, stride 1, pad 1
        call test_init_layer(layer, 3, 8, 3, 3, 1, 1)
        call random_number(layer%weights)
        call random_number(layer%bias)

        ! batch=1, 16x16 input
        allocate(input(1, 3, 16, 16))
        call random_number(input)

        call conv_forward(layer, input, output)

        out_w = (16 + 2*1 - 3) / 1 + 1  ! = 16
        out_h = (16 + 2*1 - 3) / 1 + 1  ! = 16

        if (size(output, 1) /= 1 .or. size(output, 2) /= 8 .or. &
            size(output, 3) /= 16 .or. size(output, 4) /= 16) then
            print *, "FAIL conv_forward_dimensions: expected (1,8,16,16), got", &
                     size(output,1), size(output,2), size(output,3), size(output,4)
            error stop
        end if

        print *, "PASS: conv_forward dimensions"
    end subroutine

    subroutine test_conv_forward_zero_input()
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:,:), output(:,:,:,:)
        integer :: oc, oi, oj
        real :: max_err

        call test_init_layer(layer, 2, 4, 3, 3, 1, 1)
        call random_number(layer%weights)
        layer%bias = [1.0, 2.0, 3.0, 4.0]

        allocate(input(1, 2, 8, 8))
        input = 0.0

        call conv_forward(layer, input, output)

        ! Output should equal bias at every spatial position
        max_err = 0.0
        do oc = 1, 4
            do oj = 1, size(output, 4)
                do oi = 1, size(output, 3)
                    max_err = max(max_err, abs(output(1, oc, oi, oj) - layer%bias(oc)))
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
        real, allocatable :: input(:,:,:,:), output(:,:,:,:)
        real :: expected

        ! Simplest case: 1 in, 1 out, 2x2 kernel, stride 1, no padding
        ! Input: 3x3, all ones
        ! Weights: 2x2, all ones -> each output = 4 * 1 = 4
        ! Bias: 0.5 -> each output = 4.5

        call test_init_layer(layer, 1, 1, 2, 2, 1, 0)
        layer%weights = 1.0
        layer%bias = [0.5]

        allocate(input(1, 1, 3, 3))
        input = 1.0

        call conv_forward(layer, input, output)

        ! Output should be 2x2, all 4.5
        if (size(output, 3) /= 2 .or. size(output, 4) /= 2) then
            print *, "FAIL conv_forward_known_values: wrong output size"
            error stop
        end if

        expected = 4.0 + 0.5
        if (abs(output(1,1,1,1) - expected) > 1e-5 .or. &
            abs(output(1,1,2,1) - expected) > 1e-5 .or. &
            abs(output(1,1,1,2) - expected) > 1e-5 .or. &
            abs(output(1,1,2,2) - expected) > 1e-5) then
            print *, "FAIL conv_forward_known_values: expected", expected, "got", output
            error stop
        end if

        print *, "PASS: conv_forward known values"
    end subroutine

    subroutine test_conv_forward_naive_equivalence()
        type(conv_layer) :: layer
        real, allocatable :: input(:,:,:,:), output(:,:,:,:)
        real, allocatable :: expected(:,:,:,:), padded(:,:,:)
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

        allocate(input(1, in_c, w, h))
        call random_number(input)

        ! Run conv_forward
        call conv_forward(layer, input, output)

        ! Naive implementation
        out_w = (w + 2*pad - kw) / stride + 1
        out_h = (h + 2*pad - kh) / stride + 1

        allocate(padded(in_c, w + 2*pad, h + 2*pad))
        padded = 0.0
        padded(:, pad+1:pad+w, pad+1:pad+h) = input(1, :, :, :)

        allocate(expected(1, out_c, out_w, out_h))

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
                    expected(1, oc, oi, oj) = sum_val
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
        real, allocatable :: input(:,:,:,:), output1(:,:,:,:), output2(:,:,:,:)
        real :: alpha, max_err
        integer :: oc, oi, oj

        call test_init_layer(layer, 2, 3, 3, 3, 1, 1)
        call random_number(layer%weights)
        layer%bias = 0.0  ! Zero bias for pure linearity test

        allocate(input(1, 2, 8, 8))
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
        real, allocatable :: input(:,:,:,:), output(:,:,:,:)
        real, allocatable :: grad_output(:,:,:,:), grad_input(:,:,:,:)

        call test_init_layer(layer, 3, 8, 3, 3, 1, 1)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input(1, 3, 16, 16))
        call random_number(input)

        call conv_forward(layer, input, output)

        allocate(grad_output(1, 8, 16, 16))
        call random_number(grad_output)

        call conv_backward(layer, grad_output, grad_input)

        ! Check grad_input dimensions match input
        if (size(grad_input, 1) /= 1 .or. size(grad_input, 2) /= 3 .or. &
            size(grad_input, 3) /= 16 .or. size(grad_input, 4) /= 16) then
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
        real, allocatable :: input(:,:,:,:), output(:,:,:,:)
        real, allocatable :: grad_output(:,:,:,:), grad_input(:,:,:,:)

        call test_init_layer(layer, 2, 4, 3, 3, 1, 1)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input(1, 2, 8, 8))
        call random_number(input)

        call conv_forward(layer, input, output)

        allocate(grad_output(1, 4, 8, 8))
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
        real, allocatable :: input(:,:,:,:), output(:,:,:,:), output_plus(:,:,:,:), output_minus(:,:,:,:)
        real, allocatable :: grad_output(:,:,:,:), grad_input(:,:,:,:)
        real :: eps, numerical_grad, analytical_grad, max_err, loss_plus, loss_minus
        real :: original_weight
        integer :: oc, ic, ki, kj

        eps = 1e-3
        max_err = 0.0

        call test_init_layer(layer, 2, 3, 2, 2, 1, 0)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input(1, 2, 4, 4))
        call random_number(input)

        ! Forward pass to get output shape
        call conv_forward(layer, input, output)

        ! Use ones as grad_output for simpler gradient computation
        allocate(grad_output(size(output,1), size(output,2), size(output,3), size(output,4)))
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
        real, allocatable :: input(:,:,:,:), output(:,:,:,:), output_plus(:,:,:,:), output_minus(:,:,:,:)
        real, allocatable :: grad_output(:,:,:,:), grad_input(:,:,:,:)
        real :: eps, numerical_grad, analytical_grad, max_err, loss_plus, loss_minus
        real :: original_bias
        integer :: oc

        eps = 1e-3
        max_err = 0.0

        call test_init_layer(layer, 2, 3, 2, 2, 1, 0)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input(1, 2, 4, 4))
        call random_number(input)

        call conv_forward(layer, input, output)

        allocate(grad_output(size(output,1), size(output,2), size(output,3), size(output,4)))
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
        real, allocatable :: input(:,:,:,:), output(:,:,:,:), output_plus(:,:,:,:), output_minus(:,:,:,:)
        real, allocatable :: grad_output(:,:,:,:), grad_input(:,:,:,:)
        real :: eps, numerical_grad, analytical_grad, max_err, loss_plus, loss_minus
        real :: original_input
        integer :: ic, ii, ij

        eps = 1e-3
        max_err = 0.0

        call test_init_layer(layer, 2, 3, 2, 2, 1, 0)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input(1, 2, 4, 4))
        call random_number(input)

        call conv_forward(layer, input, output)

        allocate(grad_output(size(output,1), size(output,2), size(output,3), size(output,4)))
        grad_output = 1.0

        call conv_backward(layer, grad_output, grad_input)

        ! Check a subset of input gradients
        do ic = 1, 2
            do ii = 1, 3
                do ij = 1, 3
                    original_input = input(1, ic, ii, ij)

                    input(1, ic, ii, ij) = original_input + eps
                    call conv_forward(layer, input, output_plus)
                    loss_plus = sum(output_plus)

                    input(1, ic, ii, ij) = original_input - eps
                    call conv_forward(layer, input, output_minus)
                    loss_minus = sum(output_minus)

                    input(1, ic, ii, ij) = original_input

                    numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)
                    analytical_grad = grad_input(1, ic, ii, ij)

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
        real :: input(1, 2, 3, 3), output(1, 2, 3, 3)

        input(1, 1, :, :) = reshape([-1.0, 0.0, 1.0, &
                                     -2.0, 0.5, 2.0, &
                                     -0.1, 0.0, 0.1], [3, 3])
        input(1, 2, :, :) = reshape([1.0, -1.0, 0.0, &
                                     3.0, -3.0, 0.0, &
                                     0.5, -0.5, 0.0], [3, 3])

        output = relu_forward(input)

        ! Check negatives become zero
        if (output(1, 1, 1, 1) /= 0.0 .or. output(1, 1, 1, 2) /= 0.0 .or. output(1, 1, 1, 3) /= 0.0) then
            print *, "FAIL relu_forward: negatives not zeroed"
            error stop
        end if

        ! Check positives pass through
        if (abs(output(1, 1, 3, 1) - 1.0) > 1e-6 .or. abs(output(1, 1, 3, 2) - 2.0) > 1e-6) then
            print *, "FAIL relu_forward: positives not passed"
            error stop
        end if

        ! Check zeros stay zero
        if (output(1, 1, 2, 1) /= 0.0 .or. output(1, 2, 3, 1) /= 0.0) then
            print *, "FAIL relu_forward: zeros changed"
            error stop
        end if

        print *, "PASS: relu_forward"
    end subroutine

    subroutine test_relu_backward()
        real :: x(1, 1, 4, 1), grad_out(1, 1, 4, 1), grad_in(1, 1, 4, 1)

        x(1, 1, :, 1) = [-1.0, 0.0, 1.0, 2.0]
        grad_out(1, 1, :, 1) = [5.0, 5.0, 5.0, 5.0]

        grad_in = relu_backward(x, grad_out)

        ! Negative x -> gradient blocked
        if (grad_in(1, 1, 1, 1) /= 0.0) then
            print *, "FAIL relu_backward: gradient not blocked for negative"
            error stop
        end if

        ! Zero x -> gradient blocked (x > 0 is false)
        if (grad_in(1, 1, 2, 1) /= 0.0) then
            print *, "FAIL relu_backward: gradient not blocked for zero"
            error stop
        end if

        ! Positive x -> gradient passes
        if (abs(grad_in(1, 1, 3, 1) - 5.0) > 1e-6 .or. abs(grad_in(1, 1, 4, 1) - 5.0) > 1e-6) then
            print *, "FAIL relu_backward: gradient not passed for positive"
            error stop
        end if

        print *, "PASS: relu_backward"
    end subroutine

    subroutine test_sigmoid_forward()
        real :: input(1, 1, 3, 1), output(1, 1, 3, 1)

        input(1, 1, :, 1) = [0.0, 10.0, -10.0]

        output = sigmoid_forward(input)

        ! sigmoid(0) = 0.5
        if (abs(output(1, 1, 1, 1) - 0.5) > 1e-6) then
            print *, "FAIL sigmoid_forward: sigmoid(0) /= 0.5, got", output(1, 1, 1, 1)
            error stop
        end if

        ! sigmoid(10) ~ 1
        if (abs(output(1, 1, 2, 1) - 1.0) > 1e-4) then
            print *, "FAIL sigmoid_forward: sigmoid(10) not near 1, got", output(1, 1, 2, 1)
            error stop
        end if

        ! sigmoid(-10) ~ 0
        if (abs(output(1, 1, 3, 1)) > 1e-4) then
            print *, "FAIL sigmoid_forward: sigmoid(-10) not near 0, got", output(1, 1, 3, 1)
            error stop
        end if

        print *, "PASS: sigmoid_forward"
    end subroutine

    subroutine test_sigmoid_backward_numerical()
        real, allocatable :: x(:,:,:,:), y(:,:,:,:), grad_out(:,:,:,:), grad_in(:,:,:,:)
        real :: eps, numerical_grad, analytical_grad, max_err
        real :: y_plus, y_minus, loss_plus, loss_minus
        integer :: i, j

        eps = 1e-4
        max_err = 0.0

        allocate(x(1, 1, 4, 4))
        call random_number(x)
        x = x * 4.0 - 2.0  ! range [-2, 2]

        y = sigmoid_forward(x)

        allocate(grad_out(1, 1, 4, 4))
        grad_out = 1.0

        grad_in = sigmoid_backward(y, grad_out)

        ! Numerical check
        do j = 1, 4
            do i = 1, 4
                y_plus = 1.0 / (1.0 + exp(-(x(1, 1, i, j) + eps)))
                y_minus = 1.0 / (1.0 + exp(-(x(1, 1, i, j) - eps)))
                loss_plus = y_plus
                loss_minus = y_minus

                numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)
                analytical_grad = grad_in(1, 1, i, j)

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
        real, allocatable :: input(:,:,:,:), output(:,:,:,:)

        allocate(input(1, 3, 4, 5))
        call random_number(input)

        output = upsample(input, 2)

        if (size(output, 1) /= 1 .or. size(output, 2) /= 3 .or. &
            size(output, 3) /= 8 .or. size(output, 4) /= 10) then
            print *, "FAIL upsample_dimensions: expected (1,3,8,10), got", &
                     size(output,1), size(output,2), size(output,3), size(output,4)
            error stop
        end if

        print *, "PASS: upsample dimensions"
    end subroutine

    subroutine test_upsample_values()
        real :: input(1, 1, 2, 2), output(1, 1, 4, 4)

        input(1, 1, :, :) = reshape([1.0, 2.0, 3.0, 4.0], [2, 2])

        output = upsample(input, 2)

        ! Each input value should be replicated in a 2x2 block
        ! input(1,1) = 1.0 -> output(1:2, 1:2) = 1.0
        if (output(1,1,1,1) /= 1.0 .or. output(1,1,2,1) /= 1.0 .or. &
            output(1,1,1,2) /= 1.0 .or. output(1,1,2,2) /= 1.0) then
            print *, "FAIL upsample_values: top-left block wrong"
            error stop
        end if

        ! input(2,1) = 2.0 -> output(3:4, 1:2) = 2.0
        if (output(1,1,3,1) /= 2.0 .or. output(1,1,4,1) /= 2.0 .or. &
            output(1,1,3,2) /= 2.0 .or. output(1,1,4,2) /= 2.0) then
            print *, "FAIL upsample_values: top-right block wrong"
            error stop
        end if

        ! input(1,2) = 3.0 -> output(1:2, 3:4) = 3.0
        if (output(1,1,1,3) /= 3.0 .or. output(1,1,2,3) /= 3.0 .or. &
            output(1,1,1,4) /= 3.0 .or. output(1,1,2,4) /= 3.0) then
            print *, "FAIL upsample_values: bottom-left block wrong"
            error stop
        end if

        ! input(2,2) = 4.0 -> output(3:4, 3:4) = 4.0
        if (output(1,1,3,3) /= 4.0 .or. output(1,1,4,3) /= 4.0 .or. &
            output(1,1,3,4) /= 4.0 .or. output(1,1,4,4) /= 4.0) then
            print *, "FAIL upsample_values: bottom-right block wrong"
            error stop
        end if

        print *, "PASS: upsample values"
    end subroutine

    subroutine test_upsample_backward_dimensions()
        real, allocatable :: grad_output(:,:,:,:), grad_input(:,:,:,:)

        allocate(grad_output(1, 3, 8, 10))
        call random_number(grad_output)

        grad_input = upsample_backward(grad_output, 2)

        if (size(grad_input, 1) /= 1 .or. size(grad_input, 2) /= 3 .or. &
            size(grad_input, 3) /= 4 .or. size(grad_input, 4) /= 5) then
            print *, "FAIL upsample_backward_dimensions: expected (1,3,4,5), got", &
                     size(grad_input,1), size(grad_input,2), size(grad_input,3), size(grad_input,4)
            error stop
        end if

        print *, "PASS: upsample_backward dimensions"
    end subroutine

    subroutine test_upsample_backward_values()
        real :: grad_output(1, 1, 4, 4), grad_input(1, 1, 2, 2)

        ! Set each 2x2 block to different values that sum to known results
        grad_output(1, 1, 1:2, 1:2) = 1.0  ! sum = 4
        grad_output(1, 1, 3:4, 1:2) = 2.0  ! sum = 8
        grad_output(1, 1, 1:2, 3:4) = 0.5  ! sum = 2
        grad_output(1, 1, 3:4, 3:4) = 0.25 ! sum = 1

        grad_input = upsample_backward(grad_output, 2)

        if (abs(grad_input(1,1,1,1) - 4.0) > 1e-6) then
            print *, "FAIL upsample_backward_values: (1,1) expected 4.0, got", grad_input(1,1,1,1)
            error stop
        end if

        if (abs(grad_input(1,1,2,1) - 8.0) > 1e-6) then
            print *, "FAIL upsample_backward_values: (2,1) expected 8.0, got", grad_input(1,1,2,1)
            error stop
        end if

        if (abs(grad_input(1,1,1,2) - 2.0) > 1e-6) then
            print *, "FAIL upsample_backward_values: (1,2) expected 2.0, got", grad_input(1,1,1,2)
            error stop
        end if

        if (abs(grad_input(1,1,2,2) - 1.0) > 1e-6) then
            print *, "FAIL upsample_backward_values: (2,2) expected 1.0, got", grad_input(1,1,2,2)
            error stop
        end if

        print *, "PASS: upsample_backward values"
    end subroutine

    subroutine test_upsample_roundtrip()
        ! If we upsample then backward with grad=1, each input contributes to factor^2 outputs
        ! So backward(ones, factor) should give factor^2 at each position
        real, allocatable :: input(:,:,:,:), upsampled(:,:,:,:), grad(:,:,:,:)
        integer :: factor

        factor = 3

        allocate(input(1, 2, 4, 5))
        call random_number(input)

        upsampled = upsample(input, factor)

        allocate(grad(1, 2, 4*factor, 5*factor))
        grad = 1.0

        ! backward of all-ones gives factor^2 at each input position
        ! This verifies the gradient correctly accumulates from all factor^2 outputs
        grad = upsample_backward(grad, factor)

        if (abs(grad(1,1,1,1) - real(factor*factor)) > 1e-6) then
            print *, "FAIL upsample_roundtrip: expected", factor*factor, "got", grad(1,1,1,1)
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
        real, allocatable :: input(:,:,:,:), latent(:,:,:,:), output(:,:,:,:)

        config%input_channels = 3
        config%num_layers = 3
        config%base_channels = 32
        config%max_channels = 256
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        ! Input: batch=1, 32x32x3, after 3 stride-2 layers: 4x4
        allocate(input(1, 3, 32, 32))
        call random_number(input)

        call autoencoder_forward(net, input, latent, output)

        ! Latent should be (1, max_channels, 4, 4) after 3 stride-2 downsamples
        if (size(latent, 1) /= 1 .or. size(latent, 2) /= 256 .or. &
            size(latent, 3) /= 4 .or. size(latent, 4) /= 4) then
            print *, "FAIL autoencoder_forward_dimensions: latent shape wrong", &
                     size(latent, 1), size(latent, 2), size(latent, 3), size(latent, 4)
            error stop
        end if

        ! Output should match input dimensions
        if (size(output, 1) /= 1 .or. size(output, 2) /= 3 .or. &
            size(output, 3) /= 32 .or. size(output, 4) /= 32) then
            print *, "FAIL autoencoder_forward_dimensions: output shape wrong", &
                     size(output, 1), size(output, 2), size(output, 3), size(output, 4)
            error stop
        end if

        print *, "PASS: autoencoder_forward dimensions"
    end subroutine

    subroutine test_autoencoder_forward_output_range()
        type(autoencoder_config) :: config
        type(autoencoder) :: net
        real, allocatable :: input(:,:,:,:), latent(:,:,:,:), output(:,:,:,:)
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

        allocate(input(1, 1, 16, 16))
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

    subroutine test_autoencoder_backward_numerical()
        type(autoencoder_config) :: config
        type(autoencoder) :: net
        real, allocatable :: input(:,:,:,:), target(:,:,:,:)
        real, allocatable :: latent(:,:,:,:), output(:,:,:,:)
        real, allocatable :: output_plus(:,:,:,:), output_minus(:,:,:,:)
        real, allocatable :: latent_tmp(:,:,:,:)
        real, allocatable :: grad_loss(:,:,:,:)
        real :: eps, loss, loss_plus, loss_minus
        real :: numerical_grad, analytical_grad, max_err
        real :: original_weight
        integer :: layer_idx, oc, ic, ki, kj

        eps = 1e-3
        max_err = 0.0

        ! Small network for fast test
        config%input_channels = 1
        config%num_layers = 2
        config%base_channels = 8
        config%max_channels = 32
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        ! Small input
        allocate(input(1, 1, 8, 8))
        allocate(target(1, 1, 8, 8))
        call random_number(input)
        call random_number(target)

        ! Forward pass
        call autoencoder_forward(net, input, latent, output)

        ! MSE loss gradient: d(loss)/d(output) = 2*(output - target)/n
        allocate(grad_loss, mold=output)
        grad_loss = 2.0 * (output - target) / size(output)

        ! Backward pass
        call autoencoder_backward(net, output, grad_loss)

        ! Check encoder weight gradients numerically
        layer_idx = 1
        do oc = 1, min(2, net%encoder(layer_idx)%out_channels)
            do ic = 1, net%encoder(layer_idx)%in_channels
                do ki = 1, 2
                    do kj = 1, 2
                        original_weight = net%encoder(layer_idx)%weights(oc, ic, ki, kj)

                        ! f(w + eps)
                        net%encoder(layer_idx)%weights(oc, ic, ki, kj) = original_weight + eps
                        call autoencoder_forward(net, input, latent_tmp, output_plus)
                        loss_plus = sum((output_plus - target)**2) / size(output_plus)

                        ! f(w - eps)
                        net%encoder(layer_idx)%weights(oc, ic, ki, kj) = original_weight - eps
                        call autoencoder_forward(net, input, latent_tmp, output_minus)
                        loss_minus = sum((output_minus - target)**2) / size(output_minus)

                        ! Restore
                        net%encoder(layer_idx)%weights(oc, ic, ki, kj) = original_weight

                        numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)
                        analytical_grad = net%encoder(layer_idx)%weights_grad(oc, ic, ki, kj)

                        max_err = max(max_err, abs(numerical_grad - analytical_grad))
                    end do
                end do
            end do
        end do

        ! Check decoder weight gradients numerically
        layer_idx = 1
        do oc = 1, min(2, net%decoder(layer_idx)%out_channels)
            do ic = 1, min(2, net%decoder(layer_idx)%in_channels)
                do ki = 1, 2
                    do kj = 1, 2
                        original_weight = net%decoder(layer_idx)%weights(oc, ic, ki, kj)

                        net%decoder(layer_idx)%weights(oc, ic, ki, kj) = original_weight + eps
                        call autoencoder_forward(net, input, latent_tmp, output_plus)
                        loss_plus = sum((output_plus - target)**2) / size(output_plus)

                        net%decoder(layer_idx)%weights(oc, ic, ki, kj) = original_weight - eps
                        call autoencoder_forward(net, input, latent_tmp, output_minus)
                        loss_minus = sum((output_minus - target)**2) / size(output_minus)

                        net%decoder(layer_idx)%weights(oc, ic, ki, kj) = original_weight

                        numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)
                        analytical_grad = net%decoder(layer_idx)%weights_grad(oc, ic, ki, kj)

                        max_err = max(max_err, abs(numerical_grad - analytical_grad))
                    end do
                end do
            end do
        end do

        if (max_err > 1e-2) then
            print *, "FAIL autoencoder_backward_numerical: max error =", max_err
            error stop
        end if

        print *, "PASS: autoencoder_backward numerical (max err:", max_err, ")"
    end subroutine

    subroutine test_mse_loss()
        real :: output(1, 1, 2, 2), target(1, 1, 2, 2)
        real :: loss, expected

        ! Test 1: identical output and target -> loss = 0
        output = 1.0
        target = 1.0
        loss = mse_loss(output, target)
        if (abs(loss) > 1e-6) then
            print *, "FAIL mse_loss: expected 0 for identical, got", loss
            error stop
        end if

        ! Test 2: known values
        ! output = [1, 2, 3, 4], target = [0, 0, 0, 0]
        ! errors = [1, 2, 3, 4], squared = [1, 4, 9, 16], sum = 30
        ! mse = 30 / 4 = 7.5
        output(1, 1, :, :) = reshape([1.0, 2.0, 3.0, 4.0], [2, 2])
        target = 0.0
        loss = mse_loss(output, target)
        expected = 30.0 / 4.0
        if (abs(loss - expected) > 1e-6) then
            print *, "FAIL mse_loss: expected", expected, "got", loss
            error stop
        end if

        ! Test 3: different known values
        ! output = [2, 2, 2, 2], target = [1, 1, 1, 1]
        ! errors = [1, 1, 1, 1], squared = [1, 1, 1, 1], sum = 4
        ! mse = 4 / 4 = 1.0
        output = 2.0
        target = 1.0
        loss = mse_loss(output, target)
        if (abs(loss - 1.0) > 1e-6) then
            print *, "FAIL mse_loss: expected 1.0, got", loss
            error stop
        end if

        print *, "PASS: mse_loss"
    end subroutine

    subroutine test_mse_loss_grad()
        real :: output(1, 1, 2, 2), target(1, 1, 2, 2)
        real, allocatable :: grad(:,:,:,:)
        real :: expected_grad

        ! Test 1: identical -> gradient = 0
        output = 1.0
        target = 1.0
        grad = mse_loss_grad(output, target)
        if (maxval(abs(grad)) > 1e-6) then
            print *, "FAIL mse_loss_grad: expected 0 for identical"
            error stop
        end if

        ! Test 2: correct shape
        if (size(grad, 1) /= 1 .or. size(grad, 2) /= 1 .or. &
            size(grad, 3) /= 2 .or. size(grad, 4) /= 2) then
            print *, "FAIL mse_loss_grad: wrong shape"
            error stop
        end if

        ! Test 3: known values
        ! output = [2, 2, 2, 2], target = [1, 1, 1, 1]
        ! grad = 2 * (output - target) / n = 2 * 1 / 4 = 0.5
        output = 2.0
        target = 1.0
        grad = mse_loss_grad(output, target)
        expected_grad = 2.0 * 1.0 / 4.0
        if (maxval(abs(grad - expected_grad)) > 1e-6) then
            print *, "FAIL mse_loss_grad: expected", expected_grad, "got", grad(1,1,1,1)
            error stop
        end if

        ! Test 4: varying values
        ! output = [0, 1, 2, 3], target = [1, 1, 1, 1]
        ! diff = [-1, 0, 1, 2]
        ! grad = 2 * diff / 4 = [-0.5, 0, 0.5, 1.0]
        output(1, 1, :, :) = reshape([0.0, 1.0, 2.0, 3.0], [2, 2])
        target = 1.0
        grad = mse_loss_grad(output, target)
        if (abs(grad(1,1,1,1) - (-0.5)) > 1e-6 .or. &
            abs(grad(1,1,2,1) - 0.0) > 1e-6 .or. &
            abs(grad(1,1,1,2) - 0.5) > 1e-6 .or. &
            abs(grad(1,1,2,2) - 1.0) > 1e-6) then
            print *, "FAIL mse_loss_grad: wrong gradient values"
            error stop
        end if

        print *, "PASS: mse_loss_grad"
    end subroutine

    subroutine test_sgd_update()
        type(conv_layer) :: layer
        real :: lr, expected_weight, expected_bias

        ! Create a simple layer
        layer%in_channels = 1
        layer%out_channels = 1
        layer%kernel_width = 2
        layer%kernel_height = 2
        layer%stride = 1
        layer%padding = 0

        allocate(layer%weights(1, 1, 2, 2))
        allocate(layer%bias(1))
        allocate(layer%weights_grad(1, 1, 2, 2))
        allocate(layer%bias_grad(1))

        ! Set known values
        layer%weights = 1.0
        layer%bias = 0.5
        layer%weights_grad = 0.1
        layer%bias_grad = 0.2

        lr = 0.5

        ! Update: weights = weights - lr * grad
        ! expected: 1.0 - 0.5 * 0.1 = 0.95
        ! bias: 0.5 - 0.5 * 0.2 = 0.4
        call sgd_update(layer, lr)

        expected_weight = 1.0 - 0.5 * 0.1
        expected_bias = 0.5 - 0.5 * 0.2

        if (maxval(abs(layer%weights - expected_weight)) > 1e-6) then
            print *, "FAIL sgd_update: weights wrong, expected", expected_weight, "got", layer%weights(1,1,1,1)
            error stop
        end if

        if (abs(layer%bias(1) - expected_bias) > 1e-6) then
            print *, "FAIL sgd_update: bias wrong, expected", expected_bias, "got", layer%bias(1)
            error stop
        end if

        print *, "PASS: sgd_update"
    end subroutine

    subroutine test_save_load_weights()
        type(autoencoder_config) :: config
        type(autoencoder) :: net1, net2
        character(len=*), parameter :: filename = "test_weights.bin"
        integer :: i
        real :: max_err

        config%input_channels = 1
        config%num_layers = 2
        config%base_channels = 8
        config%max_channels = 32
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        ! Create and save
        net1 = autoencoder_init(config)
        call save_weights(net1, filename)

        ! Create new net and load
        net2 = autoencoder_init(config)
        call load_weights(net2, filename)

        ! Verify weights match
        max_err = 0.0
        do i = 1, config%num_layers
            max_err = max(max_err, maxval(abs(net1%encoder(i)%weights - net2%encoder(i)%weights)))
            max_err = max(max_err, maxval(abs(net1%encoder(i)%bias - net2%encoder(i)%bias)))
            max_err = max(max_err, maxval(abs(net1%decoder(i)%weights - net2%decoder(i)%weights)))
            max_err = max(max_err, maxval(abs(net1%decoder(i)%bias - net2%decoder(i)%bias)))
        end do

        if (max_err > 1e-10) then
            print *, "FAIL save_load_weights: weights don't match, max_err =", max_err
            error stop
        end if

        ! Clean up test file
        open(unit=99, file=filename, status="old")
        close(99, status="delete")

        print *, "PASS: save_load_weights"
    end subroutine

    subroutine test_training_loss_decreases()
        type(autoencoder_config) :: config
        type(autoencoder) :: net
        real, allocatable :: images(:,:,:,:)
        real, allocatable :: latent(:,:,:,:), output(:,:,:,:)
        real :: loss_before, loss_after
        real :: lr
        integer :: epoch, i, j, k
        real, allocatable :: grad_loss(:,:,:,:)

        config%input_channels = 1
        config%num_layers = 2
        config%base_channels = 8
        config%max_channels = 32
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        ! Create small synthetic dataset: 4 images, 1 channel, 16x16
        ! Use structured patterns (gradients) instead of random noise
        allocate(images(4, 1, 16, 16))
        do i = 1, 4
            do j = 1, 16
                do k = 1, 16
                    ! Different gradient patterns for each image
                    if (i == 1) images(i, 1, j, k) = real(j) / 16.0
                    if (i == 2) images(i, 1, j, k) = real(k) / 16.0
                    if (i == 3) images(i, 1, j, k) = real(j + k) / 32.0
                    if (i == 4) images(i, 1, j, k) = real(abs(j - k)) / 16.0
                end do
            end do
        end do

        ! Compute loss before training (using batch size 1)
        loss_before = 0.0
        do i = 1, 4
            call autoencoder_forward(net, images(i:i,:,:,:), latent, output)
            loss_before = loss_before + mse_loss(output, images(i:i,:,:,:))
        end do
        loss_before = loss_before / 4.0

        ! Train for a few epochs
        lr = 0.1
        do epoch = 1, 100
            do i = 1, 4
                call autoencoder_forward(net, images(i:i,:,:,:), latent, output)
                grad_loss = mse_loss_grad(output, images(i:i,:,:,:))
                call autoencoder_backward(net, output, grad_loss)
                call sgd_update_all(net, lr)
            end do
        end do

        ! Compute loss after training
        loss_after = 0.0
        do i = 1, 4
            call autoencoder_forward(net, images(i:i,:,:,:), latent, output)
            loss_after = loss_after + mse_loss(output, images(i:i,:,:,:))
        end do
        loss_after = loss_after / 4.0

        print *, "  loss before:", loss_before
        print *, "  loss after: ", loss_after

        if (loss_after >= loss_before) then
            print *, "FAIL test_training_loss_decreases: loss did not decrease"
            error stop
        end if

        print *, "PASS: training loss decreases"
    end subroutine

    ! ============ batched tests ============

    subroutine test_conv_forward_batched()
        ! Test that conv_forward produces correct results with batch > 1
        ! by comparing batched output to single-image outputs
        type(conv_layer) :: layer
        real, allocatable :: input_batch(:,:,:,:), output_batch(:,:,:,:)
        real, allocatable :: input_single(:,:,:,:), output_single(:,:,:,:)
        integer :: batch_size, b
        real :: max_err

        batch_size = 4
        call test_init_layer(layer, 3, 8, 3, 3, 1, 1)
        call random_number(layer%weights)
        call random_number(layer%bias)

        ! Create batch of inputs
        allocate(input_batch(batch_size, 3, 16, 16))
        call random_number(input_batch)

        ! Forward pass on entire batch
        call conv_forward(layer, input_batch, output_batch)

        ! Check dimensions
        if (size(output_batch, 1) /= batch_size) then
            print *, "FAIL conv_forward_batched: batch dimension wrong"
            error stop
        end if

        ! Compare each batch element to single-image forward
        max_err = 0.0
        do b = 1, batch_size
            allocate(input_single(1, 3, 16, 16))
            input_single(1, :, :, :) = input_batch(b, :, :, :)
            call conv_forward(layer, input_single, output_single)

            max_err = max(max_err, maxval(abs(output_batch(b:b, :, :, :) - output_single)))
            deallocate(input_single, output_single)
        end do

        if (max_err > 1e-5) then
            print *, "FAIL conv_forward_batched: batch vs single mismatch, max_err =", max_err
            error stop
        end if

        print *, "PASS: conv_forward batched"
    end subroutine

    subroutine test_conv_backward_batched()
        ! Test that gradients computed on a batch equal sum of individual gradients
        type(conv_layer) :: layer
        real, allocatable :: input_batch(:,:,:,:), output_batch(:,:,:,:)
        real, allocatable :: grad_output_batch(:,:,:,:), grad_input_batch(:,:,:,:)
        real, allocatable :: input_single(:,:,:,:), output_single(:,:,:,:)
        real, allocatable :: grad_output_single(:,:,:,:), grad_input_single(:,:,:,:)
        real, allocatable :: weights_grad_sum(:,:,:,:), bias_grad_sum(:)
        integer :: batch_size, b
        real :: max_err

        batch_size = 4
        call test_init_layer(layer, 2, 4, 3, 3, 1, 1)
        call random_number(layer%weights)
        call random_number(layer%bias)

        allocate(input_batch(batch_size, 2, 8, 8))
        call random_number(input_batch)

        ! Batched forward and backward
        call conv_forward(layer, input_batch, output_batch)
        allocate(grad_output_batch(size(output_batch,1), size(output_batch,2), &
                                   size(output_batch,3), size(output_batch,4)))
        call random_number(grad_output_batch)
        call conv_backward(layer, grad_output_batch, grad_input_batch)

        ! Save batched gradients
        allocate(weights_grad_sum, mold=layer%weights_grad)
        allocate(bias_grad_sum, mold=layer%bias_grad)
        weights_grad_sum = layer%weights_grad
        bias_grad_sum = layer%bias_grad

        ! Now compute individual gradients and sum them
        layer%weights_grad = 0.0
        layer%bias_grad = 0.0
        allocate(input_single(1, 2, 8, 8))
        allocate(grad_output_single(1, size(output_batch,2), size(output_batch,3), size(output_batch,4)))

        do b = 1, batch_size
            input_single(1, :, :, :) = input_batch(b, :, :, :)
            call conv_forward(layer, input_single, output_single)
            grad_output_single(1, :, :, :) = grad_output_batch(b, :, :, :)
            call conv_backward(layer, grad_output_single, grad_input_single)

            weights_grad_sum = weights_grad_sum - layer%weights_grad
            bias_grad_sum = bias_grad_sum - layer%bias_grad
        end do

        ! Difference should be near zero
        max_err = max(maxval(abs(weights_grad_sum)), maxval(abs(bias_grad_sum)))

        if (max_err > 1e-4) then
            print *, "FAIL conv_backward_batched: gradient sum mismatch, max_err =", max_err
            error stop
        end if

        print *, "PASS: conv_backward batched"
    end subroutine

    subroutine test_autoencoder_forward_batched()
        type(autoencoder_config) :: config
        type(autoencoder) :: net
        real, allocatable :: input_batch(:,:,:,:), latent_batch(:,:,:,:), output_batch(:,:,:,:)
        real, allocatable :: input_single(:,:,:,:), latent_single(:,:,:,:), output_single(:,:,:,:)
        integer :: batch_size, b
        real :: max_err

        config%input_channels = 3
        config%num_layers = 2
        config%base_channels = 16
        config%max_channels = 64
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        batch_size = 4
        allocate(input_batch(batch_size, 3, 16, 16))
        call random_number(input_batch)

        ! Batched forward
        call autoencoder_forward(net, input_batch, latent_batch, output_batch)

        ! Check batch dimension preserved
        if (size(output_batch, 1) /= batch_size .or. size(latent_batch, 1) /= batch_size) then
            print *, "FAIL autoencoder_forward_batched: batch dimension wrong"
            error stop
        end if

        ! Compare to single-image forwards
        max_err = 0.0
        do b = 1, batch_size
            allocate(input_single(1, 3, 16, 16))
            input_single(1, :, :, :) = input_batch(b, :, :, :)
            call autoencoder_forward(net, input_single, latent_single, output_single)

            max_err = max(max_err, maxval(abs(output_batch(b:b, :, :, :) - output_single)))
            max_err = max(max_err, maxval(abs(latent_batch(b:b, :, :, :) - latent_single)))
            deallocate(input_single, latent_single, output_single)
        end do

        if (max_err > 1e-5) then
            print *, "FAIL autoencoder_forward_batched: batch vs single mismatch, max_err =", max_err
            error stop
        end if

        print *, "PASS: autoencoder_forward batched"
    end subroutine

    subroutine test_autoencoder_backward_batched()
        type(autoencoder_config) :: config
        type(autoencoder) :: net
        real, allocatable :: input_batch(:,:,:,:), latent(:,:,:,:), output_batch(:,:,:,:)
        real, allocatable :: grad_loss_batch(:,:,:,:)
        real, allocatable :: input_single(:,:,:,:), latent_tmp(:,:,:,:), output_single(:,:,:,:)
        real, allocatable :: grad_loss_single(:,:,:,:)
        real, allocatable :: enc_wgrad_batch(:,:,:,:), dec_wgrad_batch(:,:,:,:)
        integer :: batch_size, b, layer_idx
        real :: max_err

        config%input_channels = 1
        config%num_layers = 2
        config%base_channels = 8
        config%max_channels = 32
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        batch_size = 3
        allocate(input_batch(batch_size, 1, 8, 8))
        call random_number(input_batch)

        ! Batched forward + backward
        call autoencoder_forward(net, input_batch, latent, output_batch)
        allocate(grad_loss_batch, mold=output_batch)
        call random_number(grad_loss_batch)
        call autoencoder_backward(net, output_batch, grad_loss_batch)

        ! Save batched gradients
        layer_idx = 1
        allocate(enc_wgrad_batch, source=net%encoder(layer_idx)%weights_grad)
        allocate(dec_wgrad_batch, source=net%decoder(layer_idx)%weights_grad)

        ! Accumulate single-image gradients
        net%encoder(layer_idx)%weights_grad = 0.0
        net%decoder(layer_idx)%weights_grad = 0.0

        do b = 1, batch_size
            allocate(input_single(1, 1, 8, 8))
            input_single(1, :, :, :) = input_batch(b, :, :, :)
            call autoencoder_forward(net, input_single, latent_tmp, output_single)

            allocate(grad_loss_single(1, size(output_batch,2), size(output_batch,3), size(output_batch,4)))
            grad_loss_single(1, :, :, :) = grad_loss_batch(b, :, :, :)
            call autoencoder_backward(net, output_single, grad_loss_single)

            enc_wgrad_batch = enc_wgrad_batch - net%encoder(layer_idx)%weights_grad
            dec_wgrad_batch = dec_wgrad_batch - net%decoder(layer_idx)%weights_grad

            deallocate(input_single, latent_tmp, output_single, grad_loss_single)
        end do

        max_err = max(maxval(abs(enc_wgrad_batch)), maxval(abs(dec_wgrad_batch)))

        if (max_err > 1e-4) then
            print *, "FAIL autoencoder_backward_batched: gradient sum mismatch, max_err =", max_err
            error stop
        end if

        print *, "PASS: autoencoder_backward batched"
    end subroutine

    subroutine test_training_batched()
        ! Test that training with batch_size > 1 decreases loss
        type(autoencoder_config) :: config
        type(autoencoder) :: net
        real, allocatable :: images(:,:,:,:)
        real, allocatable :: latent(:,:,:,:), output(:,:,:,:)
        real :: loss_before, loss_after
        integer :: i, j, k, batch_size

        config%input_channels = 1
        config%num_layers = 2
        config%base_channels = 8
        config%max_channels = 32
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1

        net = autoencoder_init(config)

        ! Create dataset: 8 images so we can use batch_size=4
        allocate(images(8, 1, 16, 16))
        do i = 1, 8
            do j = 1, 16
                do k = 1, 16
                    images(i, 1, j, k) = sin(real(i + j) * 0.3) * cos(real(k) * 0.2) * 0.5 + 0.5
                end do
            end do
        end do

        ! Compute loss before training
        loss_before = 0.0
        do i = 1, 8
            call autoencoder_forward(net, images(i:i,:,:,:), latent, output)
            loss_before = loss_before + mse_loss(output, images(i:i,:,:,:))
        end do
        loss_before = loss_before / 8.0

        ! Train with batch_size = 4
        batch_size = 4
        call train_network(net, images, batch_size, 50, 0.1)

        ! Compute loss after training
        loss_after = 0.0
        do i = 1, 8
            call autoencoder_forward(net, images(i:i,:,:,:), latent, output)
            loss_after = loss_after + mse_loss(output, images(i:i,:,:,:))
        end do
        loss_after = loss_after / 8.0

        print *, "  batched loss before:", loss_before
        print *, "  batched loss after: ", loss_after

        if (loss_after >= loss_before) then
            print *, "FAIL test_training_batched: loss did not decrease"
            error stop
        end if

        print *, "PASS: training batched"
    end subroutine

end program
