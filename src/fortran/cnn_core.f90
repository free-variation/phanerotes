module cnn_core
    implicit none

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
 end module
