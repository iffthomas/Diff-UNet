import numpy as np
from dataset.brats_data_utils_multi_label import get_loader_brats
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import argparse
from monai.losses.dice import DiceLoss
import yaml
from pathlib import Path
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
set_determinism(123)
import os
os.chdir(Path(__file__).resolve().parent)

data_dir = "./datasets/brats2020/MICCAI_BraTS2020_TrainingData/"
logdir = "./logs_brats/diffusion_seg_all_loss_embed/"

model_save_path = os.path.join(logdir, "model")

env = "pytorch" # or env = "pytorch" if you only have one gpu.

max_epoch = 300
batch_size = 2
val_every = 10
num_gpus = 0
device = "cuda" if torch.cuda.is_available() else "cpu"


number_modality = 4
number_targets = 3 ## WT, TC, ET

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
   
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)


    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            #get a timestep t and the probabity of this timestep sampler.sample calculates this probability distribution and samples from it
            t, weight = self.sampler.sample(x.shape[0], x.device)
            
            #now we sample the noisey image and retrun noisy image, timestep t and the noise that was added
            #in q_sample 
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            #gets all embedingt of each convolutional layer block in total we have 5 embedding layers [x0,x1,x2,x3,x4]
            embeddings = self.embed_model(image)

            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

class BraTSTrainer(Trainer): #Implemeents the custom training loop for the BraTS dataset
      #at __init__ we specify what is important like the model, the optimizer, the loss and the noise scheduler
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                        sw_batch_size=1,
                                        overlap=0.25)
        self.model = DiffUNet()

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=30,
                                                  max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    
    #herer we define the training step that is performed at each iteration! we get the batch of data here our diffusion process takes place!
    def training_step(self, batch):
        image, label = self.get_input(batch)

        #we take our label mask, from batch["label"] that has shape (1,depth,height,widht) i guess
        x_start = label

        x_start = (x_start) * 2 - 1 #why are we doing this exactly when labels are only 0 or 1 ??
        #we rerange the labels: either -1 or 1 for the mask values.


        #this is where the sampling 
        #1, we sample with UniformSampler(1000), this initializes the weights
        #2 we use the sample function of the schedulesampler abc class: we get the probability distribution with probability = weight / sum(weights)
        #since all weight are the same --> all probabilites are the same
        #3 we select a random timepoint: with indices_np = np.random.choice(len(p), size=(batch_size,), p=p) according to our probability distribution p
        #4 we use the function q_sample. Here a lot happens
            #4.1 we first calculate our beta schedule. this is how much variance noise we add for each timestep t,
            #4.2 we calulculate the sqrt_alphas_cumprod multiply it with x_start
            #add sqrt_one_minus_alphas_cumprod and multiply it with the noise
            #now we have our q_sample
            #with shape: batchsize, channels_labels, depth, height, width
        #5 we have our x_t, t and noise
        #shapes are x_t: batchsize, channels_labels, depth, height, width, noise: batchsize, channels_labels, depth, height, width, t: indiceies for each image in the batch batchsize

        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")

        #we denoise the label, here we pass the noised label in, the image, and the timesteps t
        #1. use a Unet Encoder to get the embeddings
            # U net encoder has spatial dims = 3, in_channels = number_modalities, out_channels = number_targets, features = [64, 64, 128, 256, 512, 64]
            # we get embeddings with 
        #2. use a U net that accepts this embeddings; inputs are images and noised image xt
            # U net decoder has spatial dims = 3, in_channels = number_modalities + number targets, out_channels = number_targets, features = [64, 64, 128, 256, 512, 64]
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")


        #we predict various losses between our denoised label and the original label
        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return loss 
 
    def get_input(self, batch):
        #now the image has shape 2,4,96,96,96
        #now the label has shape 2,3,96,96,96

        #each channel in the label is a binary mask for the corresponding class

        image = batch["image"]
        label = batch["label"]
       
        label = label.float()
        return image, label 

    def validation_step(self, batch):
        image, label = self.get_input(batch)    
        

        #TODO: documentate this
        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()

        target = label.cpu().numpy()
        o = output[:, 1]
        t = target[:, 1] # ce
        wt = dice(o, t)
        # core
        o = output[:, 0]
        t = target[:, 0]
        tc = dice(o, t)
        # active
        o = output[:, 2]
        t = target[:, 2]
        et = dice(o, t)
        
        return [wt, tc, et]

    def validation_end(self, mean_val_outputs):
        wt, tc, et = mean_val_outputs

        self.log("wt", wt, step=self.epoch)
        self.log("tc", tc, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", (wt+tc+et)/3, step=self.epoch)

        mean_dice = (wt + tc + et) / 3
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")

        print(f"wt is {wt}, tc is {tc}, et is {et}, mean_dice is {mean_dice}")

if __name__ == "__main__":

    train_ds, val_ds, test_ds = get_loader_brats(data_dir=Path(data_dir), batch_size=batch_size, fold=0)
    print("we are here")
    
    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17751,
                            training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
