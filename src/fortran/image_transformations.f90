module image_transformations
    implicit none

    contains
        pure function transpose_image(pixels) result(pixels_out)
            real, intent(in) :: pixels(:, :, :)
            real, allocatable :: pixels_out(:, :, :)

            integer :: channels, width, height

            channels = size(pixels, 1)
            width = size(pixels, 2)
            height = size(pixels, 3)

            pixels_out = reshape(pixels, shape = [channels, height, width], order = [1, 3, 2])
        end function

        pure function flip_image_horizontal(pixels) result(pixels_out)
            real, intent(in) :: pixels(:, :, :)
            real, allocatable :: pixels_out(:, :, :)

            integer :: width

            width = size(pixels, 2)
            pixels_out = pixels(:, width:1:-1, :)
        end function
    
        pure function flip_image_vertical(pixels) result(pixels_out)
            real, intent(in) :: pixels(:, :, :)
            real, allocatable :: pixels_out(:, :, :)

            integer :: height

            height = size(pixels, 3)
            pixels_out = pixels(:, :, height:1:-1)
        end function

end module

