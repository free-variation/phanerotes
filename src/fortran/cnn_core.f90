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
        logical :: training
        real, allocatable :: input_cache(:, :,:,:)
        real, allocatable :: col_cache(:,:)
    end type


    contains
        pure function im2col(input, kernel_width, kernel_height, stride, padding)
            ! Input layout: (channels, height, width, batch)
            real, intent(in) :: input(:,:,:,:)
            integer, intent(in) :: kernel_width, kernel_height, stride, padding
            real, allocatable :: im2col(:,:)

            integer :: batch_size, num_channels, width, height
            integer :: out_width, out_height
            real, allocatable :: padded(:,:,:,:)
            integer :: b, oi, oj, col_idx, i_start, j_start, ki, kj, base_idx

            num_channels = size(input, 1)
            height = size(input, 2)
            width = size(input, 3)
            batch_size = size(input, 4)

            out_width =  (width + 2*padding - kernel_width) / stride + 1
            out_height = (height + 2*padding - kernel_height) / stride + 1

            allocate(im2col(num_channels * kernel_width * kernel_height, out_width * out_height * batch_size))
            allocate(padded(num_channels, height + 2*padding, width + 2*padding, batch_size))
            padded = 0.0
            padded(:, padding+1:padding+height, padding+1:padding+width, :) = input

            do b = 1, batch_size
                do oj = 1, out_height
                    do oi = 1, out_width
                        col_idx = (b - 1)*out_width*out_height + (oj - 1) * out_width + oi
                        i_start = (oi - 1) * stride + 1
                        j_start = (oj - 1) * stride + 1

                        do kj = 1, kernel_height
                            do ki = 1, kernel_width
                                base_idx = (kj-1)*kernel_width*num_channels + (ki-1)*num_channels + 1
                                ! Contiguous read: channels is first dimension
                                im2col(base_idx:base_idx+num_channels-1, col_idx) = &
                                    padded(:, j_start+kj-1, i_start+ki-1, b)
                            end do
                        end do
                    end do
                end do
            end do
        end function

        pure function col2im(col_form, batch_size, num_channels, width, height, kernel_width, kernel_height, stride, padding)
            ! Output layout: (channels, height, width, batch)
            real, intent(in) :: col_form(:,:)
            integer, intent(in) :: batch_size, num_channels, width, height
            integer, intent(in) :: kernel_width, kernel_height, stride, padding
            real, allocatable :: col2im(:,:,:,:)

            integer :: out_width, out_height
            integer :: b, oi, oj, col_idx, i_start, j_start, ki, kj, base_idx
            real, allocatable :: padded(:,:,:,:)

            out_width =  (width + 2*padding - kernel_width) / stride + 1
            out_height = (height + 2*padding - kernel_height) / stride + 1
            allocate(padded(num_channels, height + 2*padding, width + 2*padding, batch_size))
            padded = 0.0

            do b = 1, batch_size
                do oj = 1, out_height
                    do oi = 1, out_width
                        col_idx = (b - 1)*out_width*out_height + (oj - 1) * out_width + oi
                        i_start = (oi - 1) * stride + 1
                        j_start = (oj - 1) * stride + 1

                        do kj = 1, kernel_height
                            do ki = 1, kernel_width
                                base_idx = (kj-1)*kernel_width*num_channels + (ki-1)*num_channels + 1
                                ! Contiguous write: channels is first dimension
                                padded(:, j_start+kj-1, i_start+ki-1, b) = &
                                    padded(:, j_start+kj-1, i_start+ki-1, b) + &
                                    col_form(base_idx:base_idx+num_channels-1, col_idx)
                            end do
                        end do
                    end do
                end do
            end do

            ! strip padding
            col2im = padded(:, padding+1:padding+height, padding+1:padding+width, :)
        end function

        subroutine conv_forward(layer, input, output)
            ! Convolution via im2col + GEMM: instead of sliding a kernel across the image,
            ! we extract all patches into columns (im2col), then a single matrix multiply
            ! applies all kernels to all patches simultaneously. Much faster on modern hardware.
            ! Input/output layout: (channels, height, width, batch)
            type(conv_layer), intent(inout) :: layer
            real, intent(in) :: input(:,:,:,:)
            real, allocatable, intent(out) :: output(:,:,:,:)

            real, allocatable :: col_form(:,:)
            real, allocatable :: W(:,:)
            integer :: batch_size, m, n, k, b, col_start, col_end
            integer :: in_height, in_width, out_width, out_height
            real, allocatable :: output_matrix(:,:)
            real :: t_im2col_start, t_im2col_end, t_sgemm_start, t_sgemm_end

            in_height = size(input, 2)
            in_width = size(input, 3)
            batch_size = size(input, 4)

            ! im2col: each column is one flattened patch (all channels, kw*kh pixels)
            ! Result shape: (in_channels * kw * kh, out_w * out_h * batch)
            call cpu_time(t_im2col_start)
            col_form = im2col(input, layer%kernel_width, layer%kernel_height, layer%stride, layer%padding)
            call cpu_time(t_im2col_end)

            ! Flatten 4D weights to 2D: each row is one output filter flattened
            ! Shape: (out_channels, in_channels * kw * kh)
            W = reshape(layer%weights, [layer%out_channels, layer%in_channels * layer%kernel_width * layer%kernel_height])
            m = layer%out_channels
            k = layer%in_channels * layer%kernel_width * layer%kernel_height
            n = size(col_form, 2)

            allocate(output_matrix(layer%out_channels, n))

            ! Core operation: output = W @ col_form
            ! Each column of result is one spatial position with all output channels
            call cpu_time(t_sgemm_start)
            call sgemm("N", "N", m, n, k, 1.0, W, m, col_form, k, 0.0, output_matrix, m)
            call cpu_time(t_sgemm_end)

            print '(A,I0,A,I0,A,F6.3,A,F6.3,A)', &
                "  conv_forward [", layer%in_channels, "->", layer%out_channels, &
                "] im2col: ", (t_im2col_end - t_im2col_start) * 1000, &
                "ms, sgemm: ", (t_sgemm_end - t_sgemm_start) * 1000, "ms"

            out_width =  (in_width + 2*layer%padding - layer%kernel_width) / layer%stride + 1
            out_height = (in_height + 2*layer%padding - layer%kernel_height) / layer%stride + 1

            ! Bias is per output channel, broadcast across all spatial positions
            output_matrix = output_matrix + spread(layer%bias, 2, n)

            ! Reshape to 4D: (channels, height, width, batch)
            allocate(output(layer%out_channels, out_height, out_width, batch_size))
            do b = 1, batch_size
                col_start = (b-1)*out_width*out_height + 1
                col_end = b*out_width*out_height
                output(:, :, :, b) = reshape(output_matrix(:, col_start:col_end), &
                                              [layer%out_channels, out_height, out_width])
            end do

            ! Cache for backward pass: need original input layout and its column form
            if (layer%training) then
                layer%input_cache = input
                layer%col_cache = col_form
            end if

        end subroutine

        subroutine conv_backward(layer, grad_output, grad_input)
            ! Backpropagation through convolution. Given gradient of loss w.r.t. output,
            ! compute gradients w.r.t. weights, bias, and input (for the layer below).
            ! The math mirrors forward: since forward was matmul, backward is also matmul.
            ! Layout: (channels, height, width, batch)
            type(conv_layer), intent(inout) :: layer
            real, intent(in) :: grad_output(:,:,:,:)
            real, intent(out), allocatable :: grad_input(:,:,:,:)

            real, allocatable :: G(:,:), CG(:,:)
            integer :: out_channels, out_width, out_height
            real, allocatable :: W(:,:), WG(:,:)
            integer :: batch_size, b, col_start, col_end, k, n
            real :: t_sgemm1_start, t_sgemm1_end, t_sgemm2_start, t_sgemm2_end
            real :: t_col2im_start, t_col2im_end

            out_channels = size(grad_output, 1)
            out_height = size(grad_output, 2)
            out_width = size(grad_output, 3)
            batch_size = size(grad_output, 4)

            ! Flatten grad_output to 2D, matching the column ordering from forward pass
            allocate(G(out_channels, out_width * out_height * batch_size))
            do b = 1, batch_size
                col_start = (b-1)*out_width*out_height + 1
                col_end = b*out_width*out_height
                G(:, col_start:col_end) = reshape(grad_output(:, :, :, b), &
                                                   [out_channels, out_height*out_width])
            end do
            k = layer%in_channels * layer%kernel_width * layer%kernel_height
            n = out_width * out_height * batch_size
            W = reshape(layer%weights, [out_channels, k])

            ! Weight gradient: d(loss)/d(W) = G @ col_cache^T
            ! Each weight connects one input patch element to one output channel.
            ! Summing over all spatial positions gives the total gradient.
            allocate(WG(out_channels, k))
            call cpu_time(t_sgemm1_start)
            call sgemm('N', 'T', out_channels, k, n, 1.0, G, out_channels, layer%col_cache, k, 0.0, WG, out_channels)
            call cpu_time(t_sgemm1_end)
            layer%weights_grad = reshape(WG, [out_channels, layer%in_channels, layer%kernel_width, layer%kernel_height])

            ! Bias gradient: sum over spatial positions. Each output position contributes
            ! equally to the bias gradient for its channel.
            layer%bias_grad = sum(G, 2)

            ! Input gradient: d(loss)/d(input) = W^T @ G, then col2im to scatter back.
            ! W^T maps output gradients back to patch gradients; col2im accumulates
            ! overlapping patches (same pixel may appear in multiple patches).
            allocate(CG(k, n))
            call cpu_time(t_sgemm2_start)
            call sgemm('T', 'N', k, n, out_channels, 1.0, W, out_channels, G, out_channels, 0.0, CG, k)
            call cpu_time(t_sgemm2_end)

            call cpu_time(t_col2im_start)
            grad_input = col2im(CG, batch_size, layer%in_channels, &
                size(layer%input_cache, 3), size(layer%input_cache, 2), &
                layer%kernel_width, layer%kernel_height, &
                layer%stride, layer%padding)
            call cpu_time(t_col2im_end)

            print '(A,I0,A,I0,A,F6.3,A,F6.3,A,F6.3,A)', &
                "  conv_backward [", layer%out_channels, "->", layer%in_channels, &
                "] sgemm_w: ", (t_sgemm1_end - t_sgemm1_start) * 1000, &
                "ms, sgemm_in: ", (t_sgemm2_end - t_sgemm2_start) * 1000, &
                "ms, col2im: ", (t_col2im_end - t_col2im_start) * 1000, "ms"
        end subroutine

        pure function upsample(input, factor)
            ! Layout: (channels, height, width, batch)
            real, intent(in) :: input(:,:,:,:)
            integer, intent(in) :: factor
            real, allocatable :: upsample(:,:,:,:)

            integer :: batch_size, num_channels, width, height
            integer :: i, j, fi, fj

            num_channels = size(input, 1)
            height = size(input, 2)
            width = size(input, 3)
            batch_size = size(input, 4)

            allocate(upsample(num_channels, factor * height, factor * width, batch_size))

            do j = 1, height
                do i = 1, width
                    do fj = 1, factor
                        do fi = 1, factor
                            upsample(:, (j - 1)*factor + fj, (i - 1)*factor + fi, :) = input(:, j, i, :)
                        end do
                    end do
                end do
            end do
        end function

        pure function upsample_backward(grad_output, factor)
            ! Layout: (channels, height, width, batch)
            real, intent(in) :: grad_output(:,:,:,:)
            integer, intent(in) :: factor
            real, allocatable :: upsample_backward(:,:,:,:)

            integer :: width, height
            integer :: i, j

            height = size(grad_output, 2) / factor
            width = size(grad_output, 3) / factor

            allocate(upsample_backward(size(grad_output,1), height, width, size(grad_output,4)))

            do j = 1, height
                do i = 1, width
                    upsample_backward(:, j, i, :) = sum(sum( &
                        grad_output(:, (j-1)*factor+1:j*factor, (i-1)*factor+1:i*factor, :), dim=3), dim=2)
                end do
            end do
        end function
 end module

