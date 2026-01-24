module train
    use cnn_autoencoder
    use cnn_core

    implicit none

    contains
        pure function mse_loss(output, target)
            real, intent(in) :: output(:, :,:,:)
            real, intent(in) :: target(:, :,:,:)
            real :: mse_loss

            mse_loss = sum((output - target)**2) / size(output)
        end function

        pure function mse_loss_grad(output, target)
            real, intent(in) :: output(:, :,:,:)
            real, intent(in) :: target(:, :,:,:)
            real, allocatable :: mse_loss_grad(:, :,:,:)

            mse_loss_grad = 2.0 * (output - target) / size(output)
        end function

        pure subroutine sgd_update(layer, learning_rate)
            type(conv_layer), intent(inout) :: layer
            real, intent(in) :: learning_rate

            layer%weights = layer%weights - learning_rate * layer%weights_grad
            layer%bias = layer%bias - learning_rate * layer%bias_grad
        end subroutine

        pure subroutine sgd_update_all(net, learning_rate)
            type(autoencoder), intent(inout) :: net
            real, intent(in) :: learning_rate
            
            integer :: i

            do i = 1, net%config%num_layers
                call sgd_update(net%encoder(i), learning_rate)
                call sgd_update(net%decoder(i), learning_rate)
            end do

            do i = 1, net%config%num_layers - 1
                call sgd_update(net%skip_projection(i), learning_rate)
            end do
        end subroutine

        subroutine train_network(net, images, batch_size, num_epochs, learning_rate, dropout_rate, run_name)
            type(autoencoder), intent(inout) :: net
            real, intent(in) :: images(:,:,:,:)
            integer, intent(in) :: batch_size, num_epochs
            real, intent(in) :: learning_rate, dropout_rate
            character(*), intent(in), optional :: run_name

            integer :: epoch, batch_start, batch_end, num_samples, num_batches, batch_num
            real, allocatable :: latent(:, :,:,:), output(:, :,:,:)
            real, allocatable :: grad_loss(:, :,:,:)
            real :: total_loss
            integer(8) :: t_start, t_end, count_rate
            character(len=256) :: checkpoint_file

            call system_clock(count_rate=count_rate)
            num_samples = size(images, 4)
            num_batches = (num_samples + batch_size - 1) / batch_size

            do epoch = 1, num_epochs
                total_loss = 0.0
                batch_num = 0

                do batch_start = 1, num_samples, batch_size
                    call system_clock(t_start)
                    batch_end = min(batch_start + batch_size - 1, num_samples)
                    batch_num = batch_num + 1

                    call autoencoder_forward(net, images(:,:,:,batch_start:batch_end), dropout_rate, latent, output)

                    total_loss = total_loss + mse_loss(output, images(:,:,:,batch_start:batch_end))
                    grad_loss = mse_loss_grad(output, images(:,:,:,batch_start:batch_end))
                    call autoencoder_backward(net, output, grad_loss)
                    call sgd_update_all(net, learning_rate)

                    call system_clock(t_end)
                    print '(A,I0,A,I0,A,I0,A,F6.2,A)', &
                        "  batch ", batch_num, "/", num_batches, " (epoch ", epoch, ") ", &
                        real(t_end - t_start) / real(count_rate), "s"
                end do

                print *, "epoch", epoch, "loss:", total_loss/num_samples

                if (present(run_name)) then
                    write(checkpoint_file, '(A,A,I0,A)') trim(run_name), "_epoch_", epoch, ".bin"
                    call save_autoencoder(net, trim(checkpoint_file))
                end if
            end do
        end subroutine 

end module
