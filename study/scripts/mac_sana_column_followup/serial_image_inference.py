"""Evaluate the original image modules serially to bound MPS residency."""
import gc
import numpy as np
import torch


def generate_serial(model, vae, images, seed, sample, pixels, pil, steps):
    # Preserve the original float32 VAE path and original sampling function.
    # Only tensor residency changes; these modules have no train-time updates.
    with torch.no_grad():
        model.to('cpu')
        vae.decoder.to('cpu')
        torch.mps.empty_cache()
        vae.encoder.to('mps')
        z = (vae.encode(torch.from_numpy(np.stack([pixels(im) for im in images])).to('mps'),
                        return_dict=False)[0]*vae.config.scaling_factor).cpu()
        vae.encoder.to('cpu')
        gc.collect()
        torch.mps.empty_cache()
        model.to('mps')
        ys = sample(model, z, seed, steps)
        model.to('cpu')
        gc.collect()
        torch.mps.empty_cache()
        vae.decoder.to('mps')
        xs = vae.decode(ys.to('mps')/vae.config.scaling_factor,
                        return_dict=False)[0].cpu().numpy()
        vae.decoder.to('cpu')
        torch.mps.empty_cache()
    return [pil(x) for x in xs]
