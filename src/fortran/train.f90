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

            call sgd_update(net%latent_mu, learning_rate)
            call sgd_update(net%latent_log_var, learning_rate)
        end subroutine

        subroutine train_network(net, images, batch_size, num_epochs, learning_rate, run_name, warmup_epochs)
            type(autoencoder), intent(inout) :: net
            real, intent(in) :: images(:,:,:,:)
            integer, intent(in) :: batch_size, num_epochs
            real, intent(in) :: learning_rate
            character(*), intent(in), optional :: run_name
            integer, intent(in), optional :: warmup_epochs

            integer :: epoch, batch_start, batch_end, num_samples, num_batches, batch_num, warmup_ep
            real, allocatable :: latent_mu(:, :,:,:), latent_log_var(:, :,:,:), output(:, :,:,:)
            real, allocatable :: grad_loss(:, :,:,:)
            real :: total_loss, t_start, t_end, target_beta
            real :: mu_sum, mu_sq_sum, var_sum, mu_mean, mu_var, var_mean
            integer :: latent_count
            character(len=256) :: checkpoint_file

            target_beta = net%config%beta
            warmup_ep = 0
            if (present(warmup_epochs)) warmup_ep = warmup_epochs

            num_samples = size(images, 1)
            num_batches = (num_samples + batch_size - 1) / batch_size

            do epoch = 1, num_epochs
                if (warmup_ep > 0) then
                    net%config%beta = target_beta * min(1.0, real(epoch - 1) / real(warmup_ep))
                end if

                total_loss = 0.0
                batch_num = 0
                mu_sum = 0.0
                mu_sq_sum = 0.0
                var_sum = 0.0
                latent_count = 0

                do batch_start = 1, num_samples, batch_size
                    call cpu_time(t_start)
                    batch_end = min(batch_start + batch_size - 1, num_samples)
                    batch_num = batch_num + 1

                    call autoencoder_forward(net, images(batch_start:batch_end,:,:,:), latent_mu, latent_log_var, output)

                    if (batch_num == 1) then
                        print *, "output min/max:", minval(output), maxval(output)
                        print *, "input min/max:", minval(images(batch_start:batch_end,:,:,:)), maxval(images(batch_start:batch_end,:,:,:))
                        print *, "latent_mu min/max:", minval(latent_mu), maxval(latent_mu)
                        print *, "latent_log_var min/max:", minval(latent_log_var), maxval(latent_log_var)
                    end if

                    total_loss = total_loss + mse_loss(output, images(batch_start:batch_end,:,:,:))
                    mu_sum = mu_sum + sum(latent_mu)
                    mu_sq_sum = mu_sq_sum + sum(latent_mu**2)
                    var_sum = var_sum + sum(latent_log_var)
                    latent_count = latent_count + size(latent_mu)
                    grad_loss = mse_loss_grad(output, images(batch_start:batch_end,:,:,:))
                    call autoencoder_backward(net, output, grad_loss)

                    if (batch_num == 1) then
                        print *, "encoder(1) grad norm:", sqrt(sum(net%encoder(1)%weights_grad**2))
                        print *, "decoder(1) grad norm:", sqrt(sum(net%decoder(1)%weights_grad**2))
                    end if

                    call sgd_update_all(net, learning_rate)

                    call cpu_time(t_end)
                    print '(A,I0,A,I0,A,I0,A,F6.2,A)', &
                        "  batch ", batch_num, "/", num_batches, " (epoch ", epoch, ") ", t_end - t_start, "s"
                end do

                mu_mean = mu_sum / latent_count
                mu_var = mu_sq_sum / latent_count - mu_mean**2
                var_mean = var_sum / latent_count
                print *, "epoch", epoch, "loss:", total_loss/num_samples, "beta:", net%config%beta
                print *, "  latent_mu mean:", mu_mean, "var:", mu_var, "log_var mean:", var_mean

                if (present(run_name)) then
                    write(checkpoint_file, '(A,A,I0,A)') trim(run_name), "_epoch_", epoch, ".bin"
                    call save_weights(net, trim(checkpoint_file))
                end if
            end do
        end subroutine 

end module
