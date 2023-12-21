from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from modules.utils import *


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")
    model = load_model_from_config(config, "pre-trained/cin256-v2/model.ckpt")
    return model


def main():
    model = get_model()
    sampler = DDIMSampler(model)



    classes = [25, 187, 448, 992]  # define classes to be sampled here
    n_samples_per_class = 6

    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0  # for unconditional guidance

    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class * [1000]).to(model.device)}
            )

            for class_label in classes:
                print(
                    f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class * [class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples_per_class,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                             min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)

    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    im = Image.fromarray(grid.astype(np.uint8))
    im.show()


if __name__ == '__main__':
    main()
