program load_unload
    use :: image

    real, allocatable :: pixels(:, :, :)
    real, allocatable :: transformed_pixels(:, :, :)
    real :: M(3,3)
    real :: cx, cy
    logical :: success

    pixels = load_image("images/test1.jpg")
    print *, "Loaded image:", size(pixels, 2), "x", size(pixels, 3), "x", size(pixels, 1)

    ! Basic transforms
    transformed_pixels = transpose_image(pixels)
    call save_image("images/test1_transposed.png", transformed_pixels, success)
    print *, "Transposed:", success

    transformed_pixels = flip_image_vertical(pixels)
    call save_image("images/test1_flipped_v.png", transformed_pixels, success)
    print *, "Flipped vertical:", success

    ! Affine transforms
    cx = size(pixels, 2) / 2.0
    cy = size(pixels, 3) / 2.0

    M = rotation_matrix(cx, cy, 45.0)
    transformed_pixels = affine_transform(pixels, M)
    call save_image("images/test1_rotated_45.png", transformed_pixels, success)
    print *, "Rotated 45 degrees:", success

    M = rotation_matrix(cx, cy, 90.0)
    transformed_pixels = affine_transform(pixels, M)
    call save_image("images/test1_rotated_90.png", transformed_pixels, success)
    print *, "Rotated 90 degrees:", success

    M = translation_matrix(50.0, 25.0)
    transformed_pixels = affine_transform(pixels, M)
    call save_image("images/test1_translated.png", transformed_pixels, success)
    print *, "Translated (50, 25):", success

    ! Combined: rotate then translate
    M = matmul(translation_matrix(100.0, 0.0), rotation_matrix(cx, cy, 30.0))
    transformed_pixels = affine_transform(pixels, M)
    call save_image("images/test1_rotate_translate.png", transformed_pixels, success)
    print *, "Rotated 30 + translated:", success

    ! Zoom in (scale up by 2x around center)
    M = scale_matrix(cx, cy, 2.0, 2.0)
    transformed_pixels = affine_transform(pixels, M)
    call save_image("images/test1_zoom_in.png", transformed_pixels, success)
    print *, "Zoom in 2x:", success

    ! Zoom out (scale down by 0.5x around center)
    M = scale_matrix(cx, cy, 0.5, 0.5)
    transformed_pixels = affine_transform(pixels, M)
    call save_image("images/test1_zoom_out.png", transformed_pixels, success)
    print *, "Zoom out 0.5x:", success

end program
