module cnn_core
    implicit none

    interface
        subroutine sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            character :: transa, transb
            integer :: m, n, k, lda, ldb, ldc
            real :: alpha, beta
            real :: a(lda,*), b(ldb,*)
            real :: c(ldc,*)
        end subroutine
    end interface

    type :: conv_layer
        integer :: in_channels, out_channels
        integer :: kernel_width, kernel_height, stride, padding

        real, allocatable :: weights(:,:,:,:)
        real, allocatable :: bias(:)
        real, allocatable :: weights_grad(:,:,:,:)
        real, allocatable :: bias_grad(:)

        logical :: training
        real, allocatable :: input_cache(:,:,:,:)
        real, allocatable :: col_cache(:,:)
    end type

    type :: conv_workspace
        real, allocatable :: padded(:,:,:,:)
        real, allocatable :: col(:,:)
        real, allocatable :: out_mat(:,:)
        real, allocatable :: W_flat(:,:)
        real, allocatable :: grad_flat(:,:)
        real, allocatable :: grad_col(:,:)
    end type

    contains

        subroutine ensure_ws_2d(buf, d1, d2)
            real, allocatable, intent(inout) :: buf(:,:)
            integer, intent(in) :: d1, d2
            if (allocated(buf)) then
                if (size(buf,1) == d1 .and. size(buf,2) == d2) return
                deallocate(buf)
            end if
            allocate(buf(d1, d2))
        end subroutine

        subroutine ensure_ws_4d(buf, d1, d2, d3, d4)
            real, allocatable, intent(inout) :: buf(:,:,:,:)
            integer, intent(in) :: d1, d2, d3, d4
            if (allocated(buf)) then
                if (size(buf,1) == d1 .and. size(buf,2) == d2 .and. &
                    size(buf,3) == d3 .and. size(buf,4) == d4) return
                deallocate(buf)
            end if
            allocate(buf(d1, d2, d3, d4))
        end subroutine

        subroutine im2col_ws(input, kw, kh, stride, padding, ws)
            real, intent(in) :: input(:,:,:,:)
            integer, intent(in) :: kw, kh, stride, padding
            type(conv_workspace), intent(inout) :: ws

            integer :: nc, nh, nw, nb, out_h, out_w
            integer :: ib, oi, oj, col_idx, i_start, j_start, ki, kj, base_idx

            nc = size(input, 1)
            nh = size(input, 2)
            nw = size(input, 3)
            nb = size(input, 4)

            out_h = (nh + 2*padding - kh) / stride + 1
            out_w = (nw + 2*padding - kw) / stride + 1

            call ensure_ws_4d(ws%padded, nc, nh + 2*padding, nw + 2*padding, nb)
            call ensure_ws_2d(ws%col, nc * kw * kh, out_w * out_h * nb)

            if (padding > 0) then
                ws%padded(:, 1:padding, :, :) = 0.0
                ws%padded(:, nh+padding+1:nh+2*padding, :, :) = 0.0
                ws%padded(:, :, 1:padding, :) = 0.0
                ws%padded(:, :, nw+padding+1:nw+2*padding, :) = 0.0
            end if
            ws%padded(:, padding+1:padding+nh, padding+1:padding+nw, :) = input

            do ib = 1, nb
                do oj = 1, out_h
                    do oi = 1, out_w
                        col_idx = (ib-1)*out_w*out_h + (oj-1)*out_w + oi
                        i_start = (oi-1)*stride + 1
                        j_start = (oj-1)*stride + 1

                        do kj = 1, kh
                            do ki = 1, kw
                                base_idx = (kj-1)*kw*nc + (ki-1)*nc + 1
                                ws%col(base_idx:base_idx+nc-1, col_idx) = &
                                    ws%padded(:, j_start+kj-1, i_start+ki-1, ib)
                            end do
                        end do
                    end do
                end do
            end do
        end subroutine

        subroutine col2im_ws(nb, nc, nw, nh, kw, kh, stride, padding, ws, output)
            integer, intent(in) :: nb, nc, nw, nh, kw, kh, stride, padding
            type(conv_workspace), intent(inout) :: ws
            real, allocatable, intent(out) :: output(:,:,:,:)

            integer :: out_w, out_h
            integer :: ib, oi, oj, col_idx, i_start, j_start, ki, kj, base_idx

            out_w = (nw + 2*padding - kw) / stride + 1
            out_h = (nh + 2*padding - kh) / stride + 1

            call ensure_ws_4d(ws%padded, nc, nh + 2*padding, nw + 2*padding, nb)
            ws%padded = 0.0

            do ib = 1, nb
                do oj = 1, out_h
                    do oi = 1, out_w
                        col_idx = (ib-1)*out_w*out_h + (oj-1)*out_w + oi
                        i_start = (oi-1)*stride + 1
                        j_start = (oj-1)*stride + 1

                        do kj = 1, kh
                            do ki = 1, kw
                                base_idx = (kj-1)*kw*nc + (ki-1)*nc + 1
                                ws%padded(:, j_start+kj-1, i_start+ki-1, ib) = &
                                    ws%padded(:, j_start+kj-1, i_start+ki-1, ib) + &
                                    ws%grad_col(base_idx:base_idx+nc-1, col_idx)
                            end do
                        end do
                    end do
                end do
            end do

            output = ws%padded(:, padding+1:padding+nh, padding+1:padding+nw, :)
        end subroutine

        pure function im2col(input, kw, kh, stride, padding)
            real, intent(in) :: input(:,:,:,:)
            integer, intent(in) :: kw, kh, stride, padding
            real, allocatable :: im2col(:,:)

            real, allocatable :: padded(:,:,:,:)
            integer :: nc, nh, nw, nb, out_h, out_w
            integer :: ib, oi, oj, col_idx, i_start, j_start, ki, kj, base_idx

            nc = size(input, 1)
            nh = size(input, 2)
            nw = size(input, 3)
            nb = size(input, 4)

            out_h = (nh + 2*padding - kh) / stride + 1
            out_w = (nw + 2*padding - kw) / stride + 1

            allocate(padded(nc, nh + 2*padding, nw + 2*padding, nb))
            allocate(im2col(nc * kw * kh, out_w * out_h * nb))

            padded = 0.0
            padded(:, padding+1:padding+nh, padding+1:padding+nw, :) = input

            do ib = 1, nb
                do oj = 1, out_h
                    do oi = 1, out_w
                        col_idx = (ib-1)*out_w*out_h + (oj-1)*out_w + oi
                        i_start = (oi-1)*stride + 1
                        j_start = (oj-1)*stride + 1

                        do kj = 1, kh
                            do ki = 1, kw
                                base_idx = (kj-1)*kw*nc + (ki-1)*nc + 1
                                im2col(base_idx:base_idx+nc-1, col_idx) = &
                                    padded(:, j_start+kj-1, i_start+ki-1, ib)
                            end do
                        end do
                    end do
                end do
            end do
        end function

        pure function col2im(col, nb, nc, nw, nh, kw, kh, stride, padding)
            real, intent(in) :: col(:,:)
            integer, intent(in) :: nb, nc, nw, nh, kw, kh, stride, padding
            real, allocatable :: col2im(:,:,:,:)

            real, allocatable :: padded(:,:,:,:)
            integer :: out_w, out_h
            integer :: ib, oi, oj, col_idx, i_start, j_start, ki, kj, base_idx

            out_w = (nw + 2*padding - kw) / stride + 1
            out_h = (nh + 2*padding - kh) / stride + 1

            allocate(padded(nc, nh + 2*padding, nw + 2*padding, nb))
            padded = 0.0

            do ib = 1, nb
                do oj = 1, out_h
                    do oi = 1, out_w
                        col_idx = (ib-1)*out_w*out_h + (oj-1)*out_w + oi
                        i_start = (oi-1)*stride + 1
                        j_start = (oj-1)*stride + 1

                        do kj = 1, kh
                            do ki = 1, kw
                                base_idx = (kj-1)*kw*nc + (ki-1)*nc + 1
                                padded(:, j_start+kj-1, i_start+ki-1, ib) = &
                                    padded(:, j_start+kj-1, i_start+ki-1, ib) + &
                                    col(base_idx:base_idx+nc-1, col_idx)
                            end do
                        end do
                    end do
                end do
            end do

            col2im = padded(:, padding+1:padding+nh, padding+1:padding+nw, :)
        end function

        subroutine conv_forward(layer, input, output, ws)
            type(conv_layer), intent(inout) :: layer
            real, intent(in) :: input(:,:,:,:)
            real, allocatable, intent(out) :: output(:,:,:,:)
            type(conv_workspace), intent(inout), optional :: ws

            real, allocatable :: col(:,:), out_mat(:,:), W_flat(:,:)
            integer :: nb, nh, nw, out_h, out_w, m, n, k, i

            nh = size(input, 2)
            nw = size(input, 3)
            nb = size(input, 4)

            out_h = (nh + 2*layer%padding - layer%kernel_height) / layer%stride + 1
            out_w = (nw + 2*layer%padding - layer%kernel_width) / layer%stride + 1

            m = layer%out_channels
            k = layer%in_channels * layer%kernel_width * layer%kernel_height
            n = out_w * out_h * nb

            if (present(ws)) then
                call im2col_ws(input, layer%kernel_width, layer%kernel_height, &
                              layer%stride, layer%padding, ws)

                call ensure_ws_2d(ws%W_flat, m, k)
                ws%W_flat(1:m, 1:k) = reshape(layer%weights, [m, k])

                call ensure_ws_2d(ws%out_mat, m, n)
                call sgemm("N", "N", m, n, k, 1.0, ws%W_flat, m, ws%col, k, 0.0, ws%out_mat, m)

                do i = 1, n
                    ws%out_mat(1:m, i) = ws%out_mat(1:m, i) + layer%bias
                end do

                output = reshape(ws%out_mat(1:m, 1:n), [m, out_h, out_w, nb])

                if (layer%training) then
                    layer%input_cache = input
                    layer%col_cache = ws%col(1:k, 1:n)
                end if
            else
                col = im2col(input, layer%kernel_width, layer%kernel_height, &
                            layer%stride, layer%padding)

                allocate(W_flat(m, k))
                W_flat = reshape(layer%weights, [m, k])

                allocate(out_mat(m, n))
                call sgemm("N", "N", m, n, k, 1.0, W_flat, m, col, k, 0.0, out_mat, m)

                do i = 1, n
                    out_mat(:, i) = out_mat(:, i) + layer%bias
                end do

                output = reshape(out_mat, [m, out_h, out_w, nb])

                if (layer%training) then
                    layer%input_cache = input
                    layer%col_cache = col
                end if
            end if
        end subroutine

        subroutine conv_backward(layer, grad_output, grad_input, ws)
            type(conv_layer), intent(inout) :: layer
            real, intent(in) :: grad_output(:,:,:,:)
            real, allocatable, intent(out) :: grad_input(:,:,:,:)
            type(conv_workspace), intent(inout), optional :: ws

            real, allocatable :: grad_flat(:,:), W_flat(:,:), grad_col(:,:)
            integer :: out_C, out_h, out_w, nb, k, n, in_H, in_W

            out_C = size(grad_output, 1)
            out_h = size(grad_output, 2)
            out_w = size(grad_output, 3)
            nb = size(grad_output, 4)

            k = layer%in_channels * layer%kernel_width * layer%kernel_height
            n = out_w * out_h * nb

            in_H = size(layer%input_cache, 2)
            in_W = size(layer%input_cache, 3)

            if (present(ws)) then
                call ensure_ws_2d(ws%grad_flat, out_C, n)
                ws%grad_flat(1:out_C, 1:n) = reshape(grad_output, [out_C, n])

                call ensure_ws_2d(ws%W_flat, out_C, k)
                ws%W_flat(1:out_C, 1:k) = reshape(layer%weights, [out_C, k])

                call sgemm('N', 'T', out_C, k, n, 1.0, ws%grad_flat, out_C, &
                          layer%col_cache, k, 0.0, ws%W_flat, out_C)
                layer%weights_grad = reshape(ws%W_flat(1:out_C, 1:k), &
                    [out_C, layer%in_channels, layer%kernel_width, layer%kernel_height])

                layer%bias_grad = sum(ws%grad_flat(1:out_C, 1:n), 2)

                call ensure_ws_2d(ws%grad_col, k, n)
                call sgemm('T', 'N', k, n, out_C, 1.0, &
                          reshape(layer%weights, [out_C, k]), out_C, &
                          ws%grad_flat, out_C, 0.0, ws%grad_col, k)

                call col2im_ws(nb, layer%in_channels, in_W, in_H, &
                              layer%kernel_width, layer%kernel_height, &
                              layer%stride, layer%padding, ws, grad_input)
            else
                grad_flat = reshape(grad_output, [out_C, n])

                allocate(W_flat(out_C, k))
                W_flat = reshape(layer%weights, [out_C, k])

                call sgemm('N', 'T', out_C, k, n, 1.0, grad_flat, out_C, &
                          layer%col_cache, k, 0.0, W_flat, out_C)
                layer%weights_grad = reshape(W_flat, &
                    [out_C, layer%in_channels, layer%kernel_width, layer%kernel_height])

                layer%bias_grad = sum(grad_flat, 2)

                allocate(grad_col(k, n))
                call sgemm('T', 'N', k, n, out_C, 1.0, &
                          reshape(layer%weights, [out_C, k]), out_C, &
                          grad_flat, out_C, 0.0, grad_col, k)

                grad_input = col2im(grad_col, nb, layer%in_channels, in_W, in_H, &
                                   layer%kernel_width, layer%kernel_height, &
                                   layer%stride, layer%padding)
            end if
        end subroutine

        pure function upsample(input, factor)
            real, intent(in) :: input(:,:,:,:)
            integer, intent(in) :: factor
            real, allocatable :: upsample(:,:,:,:)

            integer :: nc, nh, nw, nb, i, j, fi, fj

            nc = size(input, 1)
            nh = size(input, 2)
            nw = size(input, 3)
            nb = size(input, 4)

            allocate(upsample(nc, nh*factor, nw*factor, nb))

            do j = 1, nh
                do i = 1, nw
                    do fj = 1, factor
                        do fi = 1, factor
                            upsample(:, (j-1)*factor+fj, (i-1)*factor+fi, :) = input(:, j, i, :)
                        end do
                    end do
                end do
            end do
        end function

        pure function upsample_backward(grad_output, factor)
            real, intent(in) :: grad_output(:,:,:,:)
            integer, intent(in) :: factor
            real, allocatable :: upsample_backward(:,:,:,:)

            integer :: nh, nw, i, j

            nh = size(grad_output, 2) / factor
            nw = size(grad_output, 3) / factor

            allocate(upsample_backward(size(grad_output,1), nh, nw, size(grad_output,4)))

            do j = 1, nh
                do i = 1, nw
                    upsample_backward(:, j, i, :) = sum(sum( &
                        grad_output(:, (j-1)*factor+1:j*factor, (i-1)*factor+1:i*factor, :), &
                        dim=3), dim=2)
                end do
            end do
        end function

end module
