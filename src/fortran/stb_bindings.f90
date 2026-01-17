module stb_bindings
    use, intrinsic :: iso_c_binding
    implicit none

    interface
        function load_image(filename, x, y, channels, desired_channels) bind(C, name = "stbi_load")
            import :: c_ptr, c_int, c_char

            character(kind = c_char), intent(in) :: filename(*)
            integer(c_int), intent(out) :: x, y, channels
            integer(c_int), value :: desired_channels

            type(c_ptr) :: load_image
        end function

       subroutine image_free(pixels) bind(C, name = "stbi_image_free")
            import :: c_ptr

            type(c_ptr), value :: pixels
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
        subroutine write_png(filename, w, h, comp, pixels, stride, success)
            character(*), intent(in) :: filename
            integer, intent(in) :: w, h, comp, stride
            type(c_ptr), intent(in) :: pixels
            logical, intent(out) :: success

            success = stbi_write_png(filename//c_null_char, w, h, comp, pixels, stride) /= 0
        end subroutine

end module stb_bindings
