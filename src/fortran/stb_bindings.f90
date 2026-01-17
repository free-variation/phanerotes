module stb_bindings
    use, intrinsic :: iso_c_binding
    implicit none

    interface
        function load_image(filename, x, y, channels, desired_channels) bind(C, name = "stb_load")
            import :: c_ptr, c_int, c_char

            character(kind = c_char), intent(in) :: filename(*)
            integer(c_int), intent(out) :: x, y, channels
            integer(c_int), value :: desired_channels

            type(c_ptr) :: load_image
        end function

        function stb_write_png(filename, w, h, comp, data, stride_in_bytes) bind(C, name = "stbi_write_png")
            import :: c_ptr, c_int, c_char

            character(kind = c_char), intent(in) :: filename(*)
            integer(c_int), value :: w, h, comp
            type(c_ptr), value :: data
            integer(c_int), value :: stride_in_bytes
            
            integer(c_int) :: stb_write_png
        end function

    end interface

    contains
        subroutine write_png(filename, w, h, comp, data, stride, success)
            character(*), intent(in) :: filename
            integer, intent(in) :: w, h, comp, stride
            type(c_ptr), intent(in) :: data
            logical, intent(out) :: success

            success = stb_write_png(filename//c_null_char, w, h, comp, data, stride) /= 0
        end subroutine

end module stb_bindings
