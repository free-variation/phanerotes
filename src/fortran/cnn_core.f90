module cnn_core
    implicit none

    interface
        subroutine sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            character :: transa, transb
            integer :: m, n, k, lda, ldb, ldc
            real :: alpha, beta
            real :: a(lda,*), b(ldb,*)
            real :: c(ldc,*)
        end subroutine
    end interface

    type :: conv_layer
        integer :: in_channels, out_channels
        integer :: kernel_width, kernel_height, stride, padding

        real, allocatable :: weights(:,:,:,:) ! out_channels, in_channels, kernel_width, kernel_height
        real, allocatable :: bias(:) ! out_channels
        real, allocatable :: weights_grad(:,:,:,:)
        real, allocatable :: bias_grad(:)

        ! cached for backward pass
        real, allocatable :: input_cache(:,:,:)
        real, allocatable :: col_cache(:,:)
    end type


    contains
        pure function im2col(input, kernel_width, kernel_height, stride, padding)
            real, intent(in) :: input(:,:,:)
            integer, intent(in) :: kernel_width, kernel_height, stride, padding
            real, allocatable :: im2col(:,:)
            
            integer :: num_channels, width, height
            integer :: out_width, out_height
            real, allocatable :: padded(:,:,:)
            integer :: oi, oj, col_idx, i_start, j_start

            num_channels = size(input, 1)
            width = size(input, 2)
            height = size(input, 3)

            out_width =  (width + 2*padding - kernel_width) / stride + 1
            out_height = (height + 2*padding -kernel_height) / stride + 1

            allocate(im2col(num_channels * kernel_width * kernel_height, out_width * out_height))
            allocate(padded(num_channels, width + 2*padding, height + 2*padding))
            padded = 0.0
            padded(:, padding+1:padding+width, padding+1:padding+height) = input

            do concurrent (oi = 1:out_width, oj = 1:out_height)
                col_idx = (oj - 1) * out_width + oi
                i_start = (oi - 1) * stride + 1
                j_start = (oj - 1) * stride + 1

                im2col(:, col_idx) = reshape(&
                    padded(:, i_start:i_start+kernel_width-1, j_start:j_start+kernel_height-1),&
                    [num_channels * kernel_width * kernel_height])
            end do
        end function

        pure function col2im(col_form, num_channels, width, height, kernel_width, kernel_height, stride, padding)
            real, intent(in) :: col_form(:,:)
            integer, intent(in) :: num_channels, width, height
            integer, intent(in) :: kernel_width, kernel_height, stride, padding
            real, allocatable :: col2im(:,:,:)

            integer :: out_width, out_height
            integer :: oi, oj, col_idx, i_start, j_start
            real, allocatable :: padded(:,:,:)

            out_width =  (width + 2*padding - kernel_width) / stride + 1
            out_height = (height + 2*padding -kernel_height) / stride + 1
            allocate(padded(num_channels, width + 2*padding, height + 2*padding))
            padded = 0.0

            do oj = 1, out_height
                do oi = 1, out_width
                    col_idx = (oj - 1) * out_width + oi
                    i_start = (oi - 1) * stride + 1
                    j_start = (oj - 1) * stride + 1

                    padded(:, i_start:i_start+kernel_width-1, j_start:j_start+kernel_height-1) = &
                        padded(:, i_start:i_start+kernel_width-1, j_start:j_start+kernel_height-1) + &
                        reshape(col_form(:, col_idx), [num_channels, kernel_width, kernel_height])
                end do
            end do

            ! strip padding
            col2im = padded(:, padding+1:padding+width, padding+1:padding+height)
            end function

        subroutine conv_forward(layer, input, output)
            ! Convolution via im2col + GEMM: instead of sliding a kernel across the image,
            ! we extract all patches into columns (im2col), then a single matrix multiply
            ! applies all kernels to all patches simultaneously. Much faster on modern hardware.
            type(conv_layer), intent(inout) :: layer
            real, intent(in) :: input(:,:,:)
            real, allocatable, intent(out) :: output(:,:,:)

            real, allocatable :: col_form(:,:)
            real, allocatable :: W(:,:)
            integer :: m, n, k
            integer :: out_width, out_height
            real, allocatable :: output_matrix(:,:)

            ! im2col: each column is one flattened patch (all channels, kw*kh pixels)
            ! Result shape: (in_channels * kw * kh, out_w * out_h)
            col_form = im2col(input, layer%kernel_width, layer%kernel_height, layer%stride, layer%padding)

            ! Flatten 4D weights to 2D: each row is one output filter flattened
            ! Shape: (out_channels, in_channels * kw * kh)
            W = reshape(layer%weights, [layer%out_channels, layer%in_channels * layer%kernel_width * layer%kernel_height])
            m = layer%out_channels
            k = layer%in_channels * layer%kernel_width * layer%kernel_height
            n = size(col_form, 2)

            allocate(output_matrix(layer%out_channels, size(col_form, 2)))

            ! Core operation: output = W @ col_form
            ! Each column of result is one spatial position with all output channels
            call sgemm("N", "N", m, n, k, 1.0, W, m, col_form, k, 0.0, output_matrix, m)

            out_width =  (size(input, 2) + 2*layer%padding - layer%kernel_width) / layer%stride + 1
            out_height = (size(input, 3) + 2*layer%padding - layer%kernel_height) / layer%stride + 1

            ! Bias is per output channel, broadcast across all spatial positions
            output_matrix = output_matrix + spread(layer%bias, 2, n)

            output = reshape(output_matrix, [layer%out_channels, out_width, out_height])

            ! Cache for backward pass: need original input layout and its column form
            layer%input_cache = input
            layer%col_cache = col_form

        end subroutine

        subroutine conv_backward(layer, grad_output, grad_input)
            ! Backpropagation through convolution. Given gradient of loss w.r.t. output,
            ! compute gradients w.r.t. weights, bias, and input (for the layer below).
            ! The math mirrors forward: since forward was matmul, backward is also matmul.
            type(conv_layer), intent(inout) :: layer
            real, intent(in) :: grad_output(:,:,:)
            real, intent(out), allocatable :: grad_input(:,:,:)

            real, allocatable :: G(:,:), CG(:,:)
            integer :: out_channels, out_width, out_height
            real, allocatable :: W(:,:), WG(:,:)
            integer :: k, n

            out_channels = size(grad_output, 1)
            out_width = size(grad_output, 2)
            out_height = size(grad_output, 3)

            ! Flatten grad_output to 2D, matching the shape used in forward pass
            G = reshape(grad_output, [out_channels, out_width * out_height])
            k = layer%in_channels * layer%kernel_width * layer%kernel_height
            n = out_width * out_height
            W = reshape(layer%weights, [out_channels, k])

            ! Weight gradient: d(loss)/d(W) = G @ col_cache^T
            ! Each weight connects one input patch element to one output channel.
            ! Summing over all spatial positions gives the total gradient.
            allocate(WG(out_channels, k))
            call sgemm('N', 'T', out_channels, k, n, 1.0, G, out_channels, layer%col_cache, k, 0.0, WG, out_channels)
            layer%weights_grad = reshape(WG, [out_channels, layer%in_channels, layer%kernel_width, layer%kernel_height])

            ! Bias gradient: sum over spatial positions. Each output position contributes
            ! equally to the bias gradient for its channel.
            layer%bias_grad = sum(G, 2)

            ! Input gradient: d(loss)/d(input) = W^T @ G, then col2im to scatter back.
            ! W^T maps output gradients back to patch gradients; col2im accumulates
            ! overlapping patches (same pixel may appear in multiple patches).
            allocate(CG(k, n))
            call sgemm('T', 'N', k, n, out_channels, 1.0, W, out_channels, G, out_channels, 0.0, CG, k)

            grad_input = col2im(CG, layer%in_channels, &
                size(layer%input_cache, 2), size(layer%input_cache, 3), &
                layer%kernel_width, layer%kernel_height, &
                layer%stride, layer%padding)
        end subroutine

        pure function upsample(input, factor)
            real, intent(in) :: input(:, :, :)
            integer, intent(in) :: factor
            real, allocatable :: upsample(:, :, :)

            integer :: num_channels, width, height
            integer :: i, j, fi, fj

            num_channels = size(input, 1)
            width = size(input, 2)
            height = size(input, 3)

            allocate(upsample(num_channels, factor * width, factor * height))

            do concurrent (i = 1:width, j = 1:height, fi = 1:factor, fj = 1:factor)
                upsample(:, (i - 1)*factor + fi, (j - 1)*factor + fj) = input(:, i, j)
            end do
        end function

        pure function upsample_backward(grad_output, factor)
            real, intent(in) :: grad_output(:,:,:)
            integer, intent(in) :: factor
            real, allocatable :: upsample_backward(:,:,:)

            integer :: num_channels, width, height
            integer :: i, j, k

            num_channels = size(grad_output, 1)
            width = size(grad_output, 2) / factor
            height = size(grad_output, 3) / factor

            allocate(upsample_backward(num_channels, width, height))

            do concurrent (i = 1:width, j = 1:height, k = 1:num_channels)
                upsample_backward(k, i, j) = sum(grad_output(k, (i - 1)*factor + 1:i*factor, (j - 1)*factor + 1:j*factor))
            end do
        end function
 end module

