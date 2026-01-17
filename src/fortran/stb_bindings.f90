module stb_bindings
    use, intrinsic :: iso_c_binding
    implicit none

    interface
        function stbi_load(filename, x, y, channels, desired_channels) bind(C, name = "stbi_load")
            import :: c_ptr, c_int, c_char

            character(kind = c_char), intent(in) :: filename(*)
            integer(c_int), intent(out) :: x, y, channels
            integer(c_int), value :: desired_channels

            type(c_ptr) :: stbi_load
        end function

       subroutine stbi_image_free(pixels_c_ptr) bind(C, name = "stbi_image_free")
            import :: c_ptr

            type(c_ptr), value :: pixels_c_ptr
        end subroutine
        
        function stbi_write_png(filename, w, h, comp, pixels, stride_in_bytes) bind(C, name = "stbi_write_png")
            import :: c_ptr, c_int, c_char

            character(kind = c_char), intent(in) :: filename(*)
            integer(c_int), value :: w, h, comp
            type(c_ptr), value :: pixels
            integer(c_int), value :: stride_in_bytes
            
            integer(c_int) :: stbi_write_png
        end function

    end interface

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

end module stb_bindings
