program check_image
    use image
    implicit none

    real, allocatable :: img(:,:,:)
    real, allocatable :: tile(:,:,:)
    integer :: c, w, h, i, j, best_i, best_j
    real :: best_max, region_max
    logical :: success

    img = load_image("images/training_data/DSCF6833.JPG")
    c = size(img, 1)
    w = size(img, 2)
    h = size(img, 3)

    print *, "Image shape:", c, w, h
    print *, "Overall min/max:", minval(img), maxval(img)
    print *, "Mean:", sum(img) / size(img)

    ! Check different regions
    print *, ""
    print *, "Top-left 128x128:"
    print *, "  min/max:", minval(img(:, 1:128, 1:128)), maxval(img(:, 1:128, 1:128))

    print *, ""
    print *, "Center 128x128:"
    print *, "  min/max:", minval(img(:, w/2:w/2+127, h/2:h/2+127)), maxval(img(:, w/2:w/2+127, h/2:h/2+127))

    print *, ""
    print *, "Sample pixels from center:"
    print *, "  pixel at (w/2, h/2):", img(:, w/2, h/2)
    print *, "  pixel at (w/2+50, h/2+50):", img(:, w/2+50, h/2+50)

    ! Find brightest region
    best_max = 0.0
    best_i = 1
    best_j = 1

    do j = 1, h - 127, 128
        do i = 1, w - 127, 128
            region_max = maxval(img(:, i:i+127, j:j+127))
            if (region_max > best_max) then
                best_max = region_max
                best_i = i
                best_j = j
            end if
        end do
    end do

    print *, ""
    print *, "Brightest 128x128 region at:", best_i, best_j
    print *, "  max value:", best_max

    ! Save tiles from brightest region
    allocate(tile(c, 128, 128))
    tile = img(:, best_i:best_i+127, best_j:best_j+127)
    print *, "  tile min/max:", minval(tile), maxval(tile)
    call save_image("bright_tile.png", tile, success)
    print *, "Saved bright_tile.png, success=", success

end program
