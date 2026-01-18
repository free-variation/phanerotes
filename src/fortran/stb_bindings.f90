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
        
        function stbi_write_bmp(filename, w, h, comp, pixels) bind(C, name = "stbi_write_bmp")
            import :: c_ptr, c_int, c_char

            character(kind = c_char), intent(in) :: filename(*)
            integer(c_int), value :: w, h, comp
            type(c_ptr), value :: pixels
            
            integer(c_int) :: stbi_write_bmp
        end function

    end interface

end module stb_bindings
