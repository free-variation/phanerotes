program load_unload
    use :: stb_bindings

    real, allocatable :: pixels(:, :, :)
    logical :: success

    pixels = load_image("images/test1.jpg" // c_null_char)
    print *, size(pixels, 2), size(pixels, 3), size(pixels, 1)

    call save_image("images/test1_copy.png", pixels, success)
    print *, success

end program
