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
            type(conv_layer), intent(inout) :: layer
            real, intent(in) :: input(:,:,:)
            real, allocatable, intent(out) :: output(:,:,:)

            real, allocatable :: col_form(:,:)
            real, allocatable :: W(:,:)
            integer :: m, n, k
            integer :: out_width, out_height
            real, allocatable :: output_matrix(:,:)

            col_form = im2col(input, layer%kernel_width, layer%kernel_height, layer%stride, layer%padding)
            W = reshape(layer%weights, [layer%out_channels, layer%in_channels * layer%kernel_width * layer%kernel_height])
            m = layer%out_channels
            k = layer%in_channels * layer%kernel_width * layer%kernel_height
            n = size(col_form, 2)

            allocate(output_matrix(layer%out_channels, size(col_form, 2)))
            
            call sgemm("N", "N", m, n, k, 1.0, W, m, col_form, k, 0.0, output_matrix, m)
            
            out_width =  (size(input, 2) + 2*layer%padding - layer%kernel_width) / layer%stride + 1
            out_height = (size(input, 3) + 2*layer%padding - layer%kernel_height) / layer%stride + 1
            output_matrix = output_matrix + spread(layer%bias, 2, n)

            output = reshape(output_matrix, [layer%out_channels, out_width, out_height])
            
            layer%input_cache = input
            layer%col_cache = col_form

        end subroutine
 end module
