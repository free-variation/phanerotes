module image
    use :: stb_bindings
    use :: stdlib_linalg, only: operator(.inv.)

    implicit none

    real, parameter :: PI = acos(-1.0)

    contains

        ! ---------- Image management ----------
        function load_image(filename) result (pixels)
            character(*), intent(in) :: filename
            real, allocatable :: pixels(:, :, :)

            type(c_ptr) :: pixels_c_ptr
            integer(c_int) :: width, height, channels
            integer(c_int8_t), pointer :: flat_pixels(:)

            pixels_c_ptr = stbi_load(trim(filename) // c_null_char, width, height, channels, 0)
            if (.not. c_associated(pixels_c_ptr)) then
                print *, "failed to load image: ", trim(filename)
                error stop
            end if
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

            success = stbi_write_bmp(trim(filename) // c_null_char, width, height, channels, &
                                    c_loc(flat_pixels)) /= 0

            deallocate(flat_pixels)
        end subroutine


        ! ---------- Image transforms ----------
        pure function transpose_image(pixels) result(pixels_out)
            real, intent(in) :: pixels(:, :, :)
            real, allocatable :: pixels_out(:, :, :)

            integer :: channels, width, height

            channels = size(pixels, 1)
            width = size(pixels, 2)
            height = size(pixels, 3)

            pixels_out = reshape(pixels, shape = [channels, height, width], order = [1, 3, 2])
        end function

        pure function flip_image_horizontal(pixels) result(pixels_out)
            real, intent(in) :: pixels(:, :, :)
            real, allocatable :: pixels_out(:, :, :)

            integer :: width

            width = size(pixels, 2)
            pixels_out = pixels(:, width:1:-1, :)
        end function
    
        pure function flip_image_vertical(pixels) result(pixels_out)
            real, intent(in) :: pixels(:, :, :)
            real, allocatable :: pixels_out(:, :, :)

            integer :: height

            height = size(pixels, 3)
            pixels_out = pixels(:, :, height:1:-1)
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
            width = size(pixels, 2)
            height = size(pixels, 3)

            allocate(out(channels, width, height))
            allocate(grid_i(width, height), grid_j(width, height))
            allocate(src_x(width, height), src_y(width, height))

            ! build coordinate grids
            do j = 1, height
                grid_i(:, j) = [(real(i), i = 1, width)]
            end do

            do i = 1, width
                grid_j(i, :) = [(real(j), j = 1, height)]
            end do

            M_inv = .inv. M
            
            ! compute source coordinates
            src_x = M_inv(1,1)*grid_i + M_inv(1,2)*grid_j + M_inv(1,3)
            src_y = M_inv(2,1)*grid_i + M_inv(2,2)*grid_j + M_inv(2,3)

            ! bilinear interpolation with edge replicate
            do concurrent (i = 1:width, j = 1:height)
                fx = src_x(i, j)
                fy = src_y(i, j)

                x0 = floor(fx); x1 = x0 + 1
                y0 = floor(fy); y1 = y0 +1
                dx = fx - x0; dy = fy - y0

                x0 = max(1, min(x0, width))
                x1 = max(1, min(x1, width))
                y0 = max(1, min(y0, height))
                y1 = max(1, min(y1, height))

                out(:, i, j) = (1-dx)*(1-dy)*pixels(:, x0, y0) &
                     +    dx *(1-dy)*pixels(:, x1, y0) &
                     + (1-dx)*   dy *pixels(:, x0, y1) &
                     +    dx *   dy *pixels(:, x1, y1)

             end do
         end function





end module

