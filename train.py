import copy
import os.path
import time
import torch
from util import util
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import datetime
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.

    opt_val = copy.deepcopy(opt)
    opt_val.isTrain = False
    opt_val.phase = 'val'
    opt_val.num_threads = 0  # test code only supports num_threads = 1
    opt_val.batch_size = 8
    opt_val.load_size = opt_val.crop_size
    opt_val.serial_batches = True  # disable data shuffling
    opt_val.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    val_dataset = create_dataset(opt_val)
    val_dataset_size = len(val_dataset)

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % train_dataset_size)
    print('The number of validation images = %d' % val_dataset_size)

    if not os.path.exists(f"runs/{opt.name}"):
        os.makedirs(f"runs/{opt.name}")
    summary_writer = SummaryWriter(f"runs/{opt.name}")
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        lr = model.optimizers[0].param_groups[0]['lr']
        summary_writer.add_scalar('lr', lr, global_step=epoch)
        print('Learning rate = %.7f' % lr)

        train_dataset.set_epoch(epoch)
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            # visualize training images
            if total_iters % opt.display_freq == 0:
                visuals = model.get_current_visuals()
                image_grid = util.grid_images([visuals], train=True)
                summary_writer.add_image('train/images', image_grid, global_step=opt.display_freq)

            total_iters += batch_size
            epoch_iter += batch_size
            iter_data_time = time.time()

        # loss summary
        losses = model.get_current_losses()
        print("Losses:", end=" ")
        for name, loss in losses.items():
            summary_writer.add_scalar(f'Loss/{name}', loss, global_step=epoch)
            print(f"{name}: {loss:.2f}", end=" ")
        print()

        # epoch summary
        epoch_time = time.time() - epoch_start_time
        times.append(epoch_time)
        time_elapsed = datetime.timedelta(seconds=sum(times))
        estimated_time_left = datetime.timedelta(seconds=int((opt.n_epochs + opt.n_epochs_decay - epoch) * time_elapsed.seconds / len(times)))
        print("Epoch %d / %d (%d%%) completed in %d sec, total iterations %d, time elapsed %s, estimated time left %s" %
              (epoch, opt.n_epochs + opt.n_epochs_decay, epoch / (opt.n_epochs + opt.n_epochs_decay) * 100,
               time.time() - epoch_start_time, total_iters, str(time_elapsed), str(estimated_time_left)))

        # validate model
        if epoch % 20 == 0:
            model.eval()

            print("Validating model at epoch %d..." % epoch)
            eval_start_time = time.time()

            all_visuals = []
            for i, data in enumerate(val_dataset):
                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results

                if i % 20 == 0:
                    all_visuals.append(visuals)

            image_grid = util.grid_images(all_visuals)
            summary_writer.add_image('validation/images', image_grid, global_step=epoch)

            print('Validation completed in %s' % datetime.timedelta(seconds=time.time() - eval_start_time))

            model.train()

        # save model
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('Model saved at epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()  # update learning rates at the end of every epoch.
        print()
