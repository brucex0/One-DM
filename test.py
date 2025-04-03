import argparse
import os
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.loader import Random_StyleIAMDataset, ContentData, generate_type
from models.unet import UNetModel
from tqdm import tqdm
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
import torch.distributed as dist
from utils.util import fix_seed
from PIL import Image
from torchvision import transforms

def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)

    """ set device """
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    load_content = ContentData()
    
    # Handle input text if provided, otherwise use text corpus
    if opt.texts:
        texts = opt.texts
        print(f"Using provided input text: {texts}")
    else:
        text_corpus = generate_type[opt.generate_type][1]
        with open(text_corpus, 'r') as _f:
            texts = _f.read().split()
        print(f"Using text corpus from: {text_corpus}")

    """setup data_loader instances"""
    if opt.style_dir:
        # Use custom style directory
        print(f"Using custom style directory: {opt.style_dir}")
        style_files = [f for f in os.listdir(opt.style_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        style_files.sort()
        
        # Load all style images
        style_images = []
        for style_file in style_files:
            img_path = os.path.join(opt.style_dir, style_file)
            img = Image.open(img_path).convert('L')
            
            # Resize while maintaining aspect ratio
            target_size = 64
            ratio = target_size / max(img.size)
            new_size = tuple([int(x * ratio) for x in img.size])
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Create a new white background image
            new_img = Image.new('L', (target_size, target_size), 255)
            
            # Calculate position to center the image
            x = (target_size - new_size[0]) // 2
            y = (target_size - new_size[1]) // 2
            
            # Paste the resized image onto the white background
            new_img.paste(img, (x, y))
            
            # Convert to tensor and normalize
            img_tensor = transforms.ToTensor()(new_img)
            img_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(img_tensor)
            style_images.append(img_tensor)
        
        # Average style images
        style_tensor = torch.stack(style_images).mean(dim=0).unsqueeze(0)
        style_tensor = style_tensor.to(device)
        
        # Create zero laplace tensor
        laplace_tensor = torch.zeros_like(style_tensor).to(device)
        
        # Create dummy width ID
        wid = [['custom_style']]
    else:
        raise ValueError("Please provide a style directory using --style_dir")

    target_dir = os.path.join(opt.save_dir, opt.generate_type)
    os.makedirs(target_dir, exist_ok=True)

    diffusion = Diffusion(device=device)

    """build model architecture"""
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM).to(device)
    
    """load pretrained one_dm model"""
    if len(opt.one_dm) > 0: 
        unet.load_state_dict(torch.load(f'{opt.one_dm}', map_location=torch.device('cpu')))
        print('load pretrained one_dm model from {}'.format(opt.one_dm))
    else:
        raise IOError('input the correct checkpoint path')
    unet.eval()

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)

    """generate the handwriting datasets"""
    for x_text in tqdm(texts, position=0, desc='Generating text'):
        text_ref = load_content.get_content(x_text)
        text_ref = text_ref.to(device).repeat(1, 1, 1, 1)
        x = torch.randn((1, 4, 8, (text_ref.shape[1]*32)//8)).to(device)
        
        if opt.sample_method == 'ddim':
            ema_sampled_images = diffusion.ddim_sample(unet, vae, 1, 
                                                    x, style_tensor, laplace_tensor, text_ref,
                                                    opt.sampling_timesteps, opt.eta)
        elif opt.sample_method == 'ddpm':
            ema_sampled_images = diffusion.ddpm_sample(unet, vae, 1, 
                                                    x, style_tensor, laplace_tensor, text_ref)
        else:
            raise ValueError('sample method is not supported')
        
        for index in range(len(ema_sampled_images)):
            im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
            image = im.convert("L")
            out_path = os.path.join(target_dir, wid[index][0])
            os.makedirs(out_path, exist_ok=True)
            image.save(os.path.join(out_path, x_text + ".png"))

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM64.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated', help='target dir for storing the generated characters')
    parser.add_argument('--one_dm', dest='one_dm', default='model_zoo/One-DM-ckpt.pt', required=False, help='pre-train model for generating')
    parser.add_argument('--generate_type', dest='generate_type', required=True, help='four generation settings:iv_s, iv_u, oov_s, oov_u')
    parser.add_argument('--texts', nargs='+', help='texts to generate in handwriting (optional)')
    parser.add_argument('--style_dir', help='directory containing style images')
    parser.add_argument('--device', type=str, default='cuda', help='device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim', help='choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    opt = parser.parse_args()
    main(opt)