module image
    use :: stb_bindings
    use :: stdlib_linalg, only: operator(.inv.)

    implicit none

    real, parameter :: PI = acos(-1.0)

    contains

        ! ---------- Image management ----------
        ! 3D layout: (channels, height, width) to match 4D layout (channels, height, width, batch)
        function load_image(filename) result (pixels)
            character(*), intent(in) :: filename
            real, allocatable :: pixels(:, :, :)

            type(c_ptr) :: pixels_c_ptr
            integer(c_int) :: width, height, channels
            integer(c_int8_t), pointer :: flat_pixels(:)
            real, allocatable :: temp(:,:,:)

            pixels_c_ptr = stbi_load(trim(filename) // c_null_char, width, height, channels, 0)
            if (.not. c_associated(pixels_c_ptr)) then
                print *, "failed to load image: ", trim(filename)
                error stop
            end if
            call c_f_pointer(pixels_c_ptr, flat_pixels, [width * height * channels])

            ! STB returns (channels, width, height), we want (channels, height, width)
            allocate(temp(channels, width, height))
            temp = reshape(iand(int(flat_pixels), 255), [channels, width, height]) / 255.0
            call stbi_image_free(pixels_c_ptr)

            ! Transpose width and height
            allocate(pixels(channels, height, width))
            pixels = reshape(temp, shape=[channels, height, width], order=[1, 3, 2])
        end function

        subroutine save_image(filename, pixels, success)
            ! Input layout: (channels, height, width)
            character(*), intent(in) :: filename
            real, intent(in) :: pixels(:, :, :)
            logical, intent(out) :: success

            integer(c_int8_t), allocatable, target :: flat_pixels(:)
            integer channels, width, height
            real, allocatable :: temp(:,:,:)

            channels = size(pixels, 1)
            height = size(pixels, 2)
            width = size(pixels, 3)

            ! Transpose back to (channels, width, height) for STB
            allocate(temp(channels, width, height))
            temp = reshape(pixels, shape=[channels, width, height], order=[1, 3, 2])

            allocate(flat_pixels(channels * width * height))
            flat_pixels = reshape(int(temp * 255.0, c_int8_t), [channels * width * height])

            success = stbi_write_bmp(trim(filename) // c_null_char, width, height, channels, &
                                    c_loc(flat_pixels)) /= 0

            deallocate(flat_pixels)
        end subroutine


        ! ---------- Image transforms ----------
        ! Layout: (channels, height, width)
        pure function transpose_image(pixels) result(pixels_out)
            real, intent(in) :: pixels(:, :, :)
            real, allocatable :: pixels_out(:, :, :)

            integer :: channels, width, height

            channels = size(pixels, 1)
            height = size(pixels, 2)
            width = size(pixels, 3)

            pixels_out = reshape(pixels, shape = [channels, width, height], order = [1, 3, 2])
        end function

        pure function flip_image_horizontal(pixels) result(pixels_out)
            real, intent(in) :: pixels(:, :, :)
            real, allocatable :: pixels_out(:, :, :)

            integer :: width

            width = size(pixels, 3)
            pixels_out = pixels(:, :, width:1:-1)
        end function

        pure function flip_image_vertical(pixels) result(pixels_out)
            real, intent(in) :: pixels(:, :, :)
            real, allocatable :: pixels_out(:, :, :)

            integer :: height

            height = size(pixels, 2)
            pixels_out = pixels(:, height:1:-1, :)
        end function

        pure function rotation_matrix(cx, cy, angle) result(M)
            real, intent(in) :: cx, cy, angle
            real :: M(3,3), rad, c, s

            rad = angle * PI / 180.0
            c = cos(rad)
            s = sin(rad)

            M(1,:) = [c, -s, cx - c*cx + s*cy]
            M(2,:) = [s,  c, cy - s*cx - c*cy]
            M(3,:) = [0.0, 0.0, 1.0]
        end function 

        pure function translation_matrix(tx, ty) result(M)
            real, intent(in) :: tx, ty
            real :: M(3,3)

            M(1,:) = [1.0, 0.0, tx]
            M(2,:) = [0.0, 1.0, ty]
            M(3,:) = [0.0, 0.0, 1.0]
        end function

        pure function scale_matrix(cx, cy, sx, sy) result(M)
            real, intent(in) :: cx, cy, sx, sy
            real :: M(3,3)

            M(1,:) = [sx, 0.0, cx - sx*cx]
            M(2,:) = [0.0, sy, cy - sy*cy]
            M(3,:) = [0.0, 0.0, 1.0]
        end function

        function affine_transform(pixels, M) result(out)
            ! Layout: (channels, height, width)
            real, intent(in) :: pixels(:,:,:)
            real, intent(in) :: M(3,3)
            real, allocatable :: out(:, :, :)

            integer :: width, height, channels
            integer :: i, j
            real :: M_inv(3, 3)
            real, allocatable :: grid_i(:, :), grid_j(:, :)
            real, allocatable :: src_x(:, :), src_y(:,:)
            real :: fx, fy, dx, dy
            integer :: x0, y0, x1, y1

            channels = size(pixels, 1)
            height = size(pixels, 2)
            width = size(pixels, 3)

            allocate(out(channels, height, width))
            allocate(grid_i(height, width), grid_j(height, width))
            allocate(src_x(height, width), src_y(height, width))

            ! build coordinate grids (j=row/height, i=col/width)
            do i = 1, width
                grid_i(:, i) = [(real(j), j = 1, height)]
            end do

            do j = 1, height
                grid_j(j, :) = [(real(i), i = 1, width)]
            end do

            M_inv = .inv. M

            ! compute source coordinates
            src_x = M_inv(1,1)*grid_j + M_inv(1,2)*grid_i + M_inv(1,3)
            src_y = M_inv(2,1)*grid_j + M_inv(2,2)*grid_i + M_inv(2,3)

            ! bilinear interpolation with edge replicate
            do concurrent (j = 1:height, i = 1:width)
                fx = src_x(j, i)
                fy = src_y(j, i)

                x0 = floor(fx); x1 = x0 + 1
                y0 = floor(fy); y1 = y0 + 1
                dx = fx - x0; dy = fy - y0

                x0 = max(1, min(x0, width))
                x1 = max(1, min(x1, width))
                y0 = max(1, min(y0, height))
                y1 = max(1, min(y1, height))

                out(:, j, i) = (1-dx)*(1-dy)*pixels(:, y0, x0) &
                     +    dx *(1-dy)*pixels(:, y0, x1) &
                     + (1-dx)*   dy *pixels(:, y1, x0) &
                     +    dx *   dy *pixels(:, y1, x1)

             end do
         end function





end module

