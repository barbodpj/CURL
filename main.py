
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf
'''
from data import IE_Dataset
import torch
import argparse
import torch.optim as optim
import numpy as np
import datetime
import os.path
import os
import NEW_MODEL
import sys
from torch.utils.tensorboard import SummaryWriter
np.set_printoptions(threshold=sys.maxsize)
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
import cv2

def main():
    PSNR = PeakSignalNoiseRatio(data_range=1, dim=(1, 2, 3), reduction="sum")
    SSIM = StructuralSimilarityIndexMeasure(reduction="sum")

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join('./runs', timestamp)
    writer = SummaryWriter(run_dir)

    parser = argparse.ArgumentParser(
        description="Train the CURL neural network on image pairs")

    parser.add_argument(
        "--num_epoch", type=int, required=False, help="Number of epoches (default 5000)", default=100000)
    parser.add_argument(
        "--valid_every", type=int, required=False, help="Number of epoches after which to compute validation accuracy",
        default=1)
    parser.add_argument(
        "--checkpoint_filepath", required=False, help="Location of checkpoint file", default="a.pt")
    parser.add_argument(
        "--val_path", required=False,
        help="Directory containing images to run through a saved CURL model instance", default="./dataset/val")
    parser.add_argument(
        "--train_path", required=False,
        help="Directory containing images to train a DeepLPF model instance", default="./dataset/train")
    parser.add_argument(
        "--test_path", required=False,
        help="Directory containing images to train a DeepLPF model instance", default="./dataset/test")
    parser.add_argument("--device", required=False, help="Cuda or CPU", default="cuda")
    parser.add_argument("--mode", required=False, help="Train or Test", default="test")
    parser.add_argument("--batch_size", required=False, help="Batch Size", default=2)


    args = parser.parse_args()
    num_epoch = args.num_epoch
    valid_every = args.valid_every
    mode = args.mode
    checkpoint_filepath = args.checkpoint_filepath
    val_dir_path = args.val_path
    training_img_dirpath = args.train_path
    test_dirpath = args.test_path
    DEVICE= args.device
    BATCH_SIZE = args.batch_size




    if mode=="train":
        val_dataset = IE_Dataset(data_dir=val_dir_path, target_size=(1000, 600))
        train_dataset = IE_Dataset(data_dir=training_img_dirpath, target_size=(1000, 600))
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    if mode == "test":
        test_dataset = IE_Dataset(data_dir=test_dirpath, target_size=(1500, 1000))
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    net = NEW_MODEL.CURLNet()
    net.to(device=DEVICE)
    if checkpoint_filepath is not None:
        checkpoint = torch.load(checkpoint_filepath, map_location=DEVICE)
        print("Loading from saved model")
        net.load_state_dict(checkpoint['model_state_dict'],strict=False)

    criterion = NEW_MODEL.NEW_CURLLoss()


    start_epoch=0

    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                              net.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-10)

    total_examples = 0

    if mode=="train":
        print("Training Started")
        for epoch in range(start_epoch,num_epoch):
            print("Epoch: " , epoch)
            net.train()
            examples = 0.0
            running_loss = 0.0
            for batch_num, data in enumerate(train_data_loader, 0):

                input_img_batch, gt_img_batch,_ = data
                if DEVICE == 'cuda':
                    input_img_batch = input_img_batch.cuda()
                    gt_img_batch = gt_img_batch.cuda()

                net_img_batch, gradient_regulariser = net(
                    input_img_batch)
                net_img_batch = torch.clamp(
                    net_img_batch, 0.0, 1.0)

                loss = criterion(net_img_batch,gt_img_batch, gradient_regulariser)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                examples += BATCH_SIZE
                total_examples+=BATCH_SIZE

                writer.add_scalar('Train/Loss', loss.data[0], total_examples)
                writer.flush()


            running_loss/=examples
            writer.add_scalar('Train/Loss_Smooth', running_loss, epoch + 1)
            print("Loss_Smooth in Epoch " , epoch , "is " , running_loss)
            if (epoch + 1) % valid_every == 0:
                print("Validation Started ")
                net.eval()
                examples = 0
                running_loss = 0
                total_ssim = 0
                total_psnr = 0
                with torch.no_grad():
                    for batch_num, data in enumerate(val_data_loader, 0):

                        input_img_batch, gt_img_batch,file_name = data
                        if DEVICE == 'cuda':
                            input_img_batch = input_img_batch.cuda()
                            gt_img_batch = gt_img_batch.cuda()

                        net_img_batch, gradient_regulariser = net(
                            input_img_batch)
                        net_img_batch = torch.clamp(
                            net_img_batch, 0.0, 1.0)


                        loss = criterion(net_img_batch,
                                         gt_img_batch, gradient_regulariser)

                        running_loss += loss.data[0]
                        total_psnr += PSNR(net_img_batch,gt_img_batch)
                        total_ssim += SSIM(net_img_batch, gt_img_batch)

                        for ind in range(net_img_batch.shape[0]):
                            writer.add_image('Results on ' + file_name[ind], net_img_batch[ind][:,:,::-1], epoch+1,
                                             dataformats='CHW')
                        examples += BATCH_SIZE


                running_loss/=examples
                writer.add_scalar('Valid/Loss_Smooth', running_loss, epoch + 1)
                writer.add_scalar('Valid/PSNR', total_psnr / examples, epoch + 1)
                writer.add_scalar('Valid/SSIM', total_ssim / examples, epoch + 1)
                writer.flush()

                torch.save({
                    'epoch': epoch+1,
                     'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                     'loss': running_loss,
                     }, run_dir+"/epoch" + str(epoch+1) + "_model.pth")

    if mode=="test":
        print("Testing Started")
        net.eval()
        with torch.no_grad():

            for batch_num, data in enumerate(test_data_loader, 0):
                input_img_batch, gt_img_batch,file_name = data
                if DEVICE == 'cuda':
                    input_img_batch = input_img_batch.cuda()
                    gt_img_batch = gt_img_batch.cuda()

                net_img_batch, gradient_regulariser = net(
                    input_img_batch)
                net_img_batch = torch.clamp(
                    net_img_batch, 0.0, 1.0)

                loss = criterion(net_img_batch,
                                 gt_img_batch, gradient_regulariser)

                writer.add_scalar('Test/Loss_Smooth', loss.data[0], batch_num)
                writer.add_scalar('Test/PSNR',  PSNR(net_img_batch, gt_img_batch), batch_num)
                writer.add_scalar('Test/SSIM', SSIM(net_img_batch, gt_img_batch), batch_num)

                for ind in range(net_img_batch.shape[0]):
                    writer.add_image('Results on ' + file_name[ind], torch.flip(net_img_batch[ind],dims=(0,)),1,
                                     dataformats='CHW')
                    image = net_img_batch[ind].permute(1,2,0).cpu().numpy()
                    image = image*255
                    image = image.astype('uint8')
                    cv2.imwrite("results/" + file_name[ind],image)
            writer.flush()



if __name__ == "__main__":
    main()
