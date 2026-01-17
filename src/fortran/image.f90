module image
    use :: stb_bindings
    implicit none

    contains
        function load_image(filename) result (pixels)
            character(*), intent(in) :: filename
            real, allocatable :: pixels(:, :, :)

            type(c_ptr) :: pixels_c_ptr
            integer(c_int) :: width, height, channels
            integer(c_int8_t), pointer :: flat_pixels(:)

            pixels_c_ptr = stbi_load(filename // c_null_char, width, height, channels, 0)
            call c_f_pointer(pixels_c_ptr, flat_pixels, [width * height * channels])

            allocate(pixels(channels, width, height))
            pixels = reshape(iand(int(flat_pixels), 255), [channels, width, height]) / 255.0

            call stbi_image_free(pixels_c_ptr)
        end function

        subroutine save_image(filename, pixels, success)
            character(*), intent(in) :: filename
            real, intent(in) :: pixels(:, :, :)
            logical, intent(out) :: success

            integer(c_int8_t), allocatable, target :: flat_pixels(:)
            integer channels, width, height

            channels = size(pixels, 1)
            width = size(pixels, 2)
            height = size(pixels, 3)

            allocate(flat_pixels(channels * width * height))
            flat_pixels = reshape(int(pixels * 255.0, c_int8_t), [channels * width * height])

            success = stbi_write_png(filename // c_null_char, width, height, channels, &
                                    c_loc(flat_pixels), width * channels) /= 0

            deallocate(flat_pixels)
        end subroutine


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

