program load_unload
    use :: image

    real, allocatable :: pixels(:, :, :)
    logical :: success
    real, allocatable :: transformed_pixels(:, :, :)

    pixels = load_image("images/test1.jpg")
    print *, size(pixels, 2), size(pixels, 3), size(pixels, 1)

    transformed_pixels = transpose_image(pixels)
    call save_image("images/test1_transposed.png", transformed_pixels, success)
    print *, success
    
    transformed_pixels = flip_image_vertical(pixels)
    call save_image("images/test1_flipped_v.png", transformed_pixels, success)
    print *, success

end program
