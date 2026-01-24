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
        subroutine im2col(input, kernel_width, kernel_height, stride, padding, col_out)
            real, intent(in) :: input(:,:,:,:)
            integer, intent(in) :: kernel_width, kernel_height, stride, padding
            real, allocatable, intent(out) :: col_out(:,:)

            integer :: nc, nh, nw, nb, out_w, out_h
            real, allocatable :: padded(:,:,:,:)
            integer :: ib, oi, oj, col_idx, i_start, j_start, ki, kj, base_idx

            nc = size(input, 1)
            nh = size(input, 2)
            nw = size(input, 3)
            nb = size(input, 4)

            out_h = (nh + 2*padding - kernel_height) / stride + 1
            out_w = (nw + 2*padding - kernel_width) / stride + 1

            allocate(col_out(nc * kernel_width * kernel_height, out_w * out_h * nb))
            allocate(padded(nc, nh + 2*padding, nw + 2*padding, nb))
            padded = 0.0
            padded(:, padding+1:padding+nh, padding+1:padding+nw, :) = input

            !$omp parallel do default(shared) collapse(3) &
            !$omp& private(ib, oj, oi, col_idx, i_start, j_start, kj, ki, base_idx)
            do ib = 1, nb
                do oj = 1, out_h
                    do oi = 1, out_w
                        col_idx = (ib - 1)*out_w*out_h + (oj - 1) * out_w + oi
                        i_start = (oi - 1) * stride + 1
                        j_start = (oj - 1) * stride + 1

                        do kj = 1, kernel_height
                            do ki = 1, kernel_width
                                base_idx = (kj-1)*kernel_width*nc + (ki-1)*nc + 1
                                col_out(base_idx:base_idx+nc-1, col_idx) = &
                                    padded(:, j_start+kj-1, i_start+ki-1, ib)
                            end do
                        end do
                    end do
                end do
            end do
            !$omp end parallel do
        end subroutine

        subroutine col2im(col_form, nb, nc, nw, nh, kernel_width, kernel_height, stride, padding, img_out)
            real, intent(in) :: col_form(:,:)
            integer, intent(in) :: nb, nc, nw, nh
            integer, intent(in) :: kernel_width, kernel_height, stride, padding
            real, allocatable, intent(out) :: img_out(:,:,:,:)

            integer :: out_w, out_h
            integer :: ib, oi, oj, col_idx, i_start, j_start, ki, kj, base_idx
            real, allocatable :: padded(:,:,:,:)

            out_w = (nw + 2*padding - kernel_width) / stride + 1
            out_h = (nh + 2*padding - kernel_height) / stride + 1
            allocate(padded(nc, nh + 2*padding, nw + 2*padding, nb))
            padded = 0.0

            !$omp parallel do default(shared) &
            !$omp& private(ib, oj, oi, col_idx, i_start, j_start, kj, ki, base_idx)
            do ib = 1, nb
                do oj = 1, out_h
                    do oi = 1, out_w
                        col_idx = (ib - 1)*out_w*out_h + (oj - 1) * out_w + oi
                        i_start = (oi - 1) * stride + 1
                        j_start = (oj - 1) * stride + 1

                        do kj = 1, kernel_height
                            do ki = 1, kernel_width
                                base_idx = (kj-1)*kernel_width*nc + (ki-1)*nc + 1
                                padded(:, j_start+kj-1, i_start+ki-1, ib) = &
                                    padded(:, j_start+kj-1, i_start+ki-1, ib) + &
                                    col_form(base_idx:base_idx+nc-1, col_idx)
                            end do
                        end do
                    end do
                end do
            end do
            !$omp end parallel do

            img_out = padded(:, padding+1:padding+nh, padding+1:padding+nw, :)
        end subroutine

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

            in_height = size(input, 2)
            in_width = size(input, 3)
            batch_size = size(input, 4)

            call im2col(input, layer%kernel_width, layer%kernel_height, layer%stride, layer%padding, col_form)

            W = reshape(layer%weights, [layer%out_channels, layer%in_channels * layer%kernel_width * layer%kernel_height])
            m = layer%out_channels
            k = layer%in_channels * layer%kernel_width * layer%kernel_height
            n = size(col_form, 2)

            allocate(output_matrix(layer%out_channels, n))
            call sgemm("N", "N", m, n, k, 1.0, W, m, col_form, k, 0.0, output_matrix, m)

            out_width =  (in_width + 2*layer%padding - layer%kernel_width) / layer%stride + 1
            out_height = (in_height + 2*layer%padding - layer%kernel_height) / layer%stride + 1

            ! Bias is per output channel, broadcast across all spatial positions
            output_matrix = output_matrix + spread(layer%bias, 2, n)

            ! Reshape to 4D: (channels, height, width, batch)
            ! im2col orders columns with width varying fastest, so use order=[1,3,2]
            allocate(output(layer%out_channels, out_height, out_width, batch_size))
            do b = 1, batch_size
                col_start = (b-1)*out_width*out_height + 1
                col_end = b*out_width*out_height
                output(:, :, :, b) = reshape(output_matrix(:, col_start:col_end), &
                                              [layer%out_channels, out_height, out_width], order=[1,3,2])
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

            out_channels = size(grad_output, 1)
            out_height = size(grad_output, 2)
            out_width = size(grad_output, 3)
            batch_size = size(grad_output, 4)

            allocate(G(out_channels, out_width * out_height * batch_size))
            do b = 1, batch_size
                col_start = (b-1)*out_width*out_height + 1
                col_end = b*out_width*out_height
                ! Match im2col's width-fastest ordering: transpose h/w then flatten
                G(:, col_start:col_end) = reshape( &
                    reshape(grad_output(:, :, :, b), [out_channels, out_width, out_height], order=[1,3,2]), &
                    [out_channels, out_width*out_height])
            end do
            k = layer%in_channels * layer%kernel_width * layer%kernel_height
            n = out_width * out_height * batch_size
            W = reshape(layer%weights, [out_channels, k])

            allocate(WG(out_channels, k))
            call sgemm('N', 'T', out_channels, k, n, 1.0, G, out_channels, layer%col_cache, k, 0.0, WG, out_channels)
            layer%weights_grad = reshape(WG, [out_channels, layer%in_channels, layer%kernel_width, layer%kernel_height])

            layer%bias_grad = sum(G, 2)

            allocate(CG(k, n))
            call sgemm('T', 'N', k, n, out_channels, 1.0, W, out_channels, G, out_channels, 0.0, CG, k)

            call col2im(CG, batch_size, layer%in_channels, &
                size(layer%input_cache, 3), size(layer%input_cache, 2), &
                layer%kernel_width, layer%kernel_height, &
                layer%stride, layer%padding, grad_input)
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

