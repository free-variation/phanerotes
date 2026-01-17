program load_unload
    use :: stb_bindings
    use, intrinsic :: iso_c_binding

    integer(c_int) x, y, channels
    type(c_ptr) pixels 
    logical success

    pixels = load_image("images/test1.jpg" // c_null_char, x, y, channels, 0)
    print *, x, y, channels

    call write_png("images/test1_copy.png", x, y, channels, pixels, x * channels, success)
    print *, success

    call image_free(pixels)
end program
