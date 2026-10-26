"""One-step image diffusion adaptation with supervision through the frozen VAE decoder."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


class DecoderCheckpoint(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, value):
        if torch.is_grad_enabled() and value.requires_grad:
            return checkpoint(self.block, value, use_reentrant=False)
        return self.block(value)


def checkpoint_decoder(decoder):
    for index, block in enumerate(decoder.up_blocks):
        decoder.up_blocks[index] = DecoderCheckpoint(block).eval()
    return decoder


def move_optimizer_moments(optimizer, device):
    for state in optimizer.state.values():
        assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, set(state)
        for key in ['exp_avg', 'exp_avg_sq']:
            state[key] = state[key].to(device)


def pixel_vjp_surrogate(clean, gradient):
    return (clean*gradient.detach()).sum()


def example_indices(rows, config, *, development=False, copy=None):
    """One data selection rule for decimal, binary and appearance curricula."""
    if development:
        if 'font_grid' in config:
            return [i for i, row in enumerate(rows) if row['split'] == 'development']
        if 'binary' in config:
            return list(range(len(rows)))
        return [i for i, row in enumerate(rows) if row['variant'] == 0 and
                (row['copy'] or row['a'] in [2, 9])]
    return [i for i, row in enumerate(rows) if row['copy'] == copy and
            ('font_grid' not in config or row['split'] == 'train')]


def semantic_loss(decoded, answer_pixels, answers, config, critic=None):
    """Training-only losses; no recognizer participates in sampling."""
    assert not ('critic' in config and 'contrast' in config)
    if 'critic' in config:
        from ocr_critic import critic_loss
        return config['critic']['weight'] * critic_loss(decoded, answers, critic)
    if 'contrast' in config:
        from pixel_contrast import digit_contrast_loss
        return config['contrast']['weight'] * digit_contrast_loss(
            decoded, answer_pixels, answers, config['contrast']['temperature'])
    return decoded.new_zeros(())


def load_pixel_model(root):
    # Imported from the manifested original trainer; preserves original pretrained weights.
    from train import load_model
    model, adapters = load_model(root)
    for name, parameter in model.named_parameters():
        if (name.startswith('patch_embed.') or name.endswith('scale_shift_table') or
                '.ff.conv_depth.' in name or
                any(name.startswith(f'transformer_blocks.{i}.ff.') for i in range(24, 28))):
            parameter.requires_grad_(True)
    # This input adapter is linear and acts on each latent patch independently.
    # The frozen VAE has already formed contextual image features; SANA then
    # processes their sequence. The projection itself mixes no latent patches.
    model.raster_conditioner = torch.nn.Linear(34, 2304, bias=False).to('mps')
    return model, adapters


def raster_tokens(context):
    assert context.shape[1:] == (32, 16, 16)
    batch = context.shape[0]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 16, device=context.device)[:9],
                            torch.linspace(-1, 1, 16, device=context.device), indexing='ij')
    coords = torch.stack([xx, yy], -1).reshape(1, 144, 2).expand(batch, -1, -1)
    image_tokens = context[:, :, :9, :].permute(0, 2, 3, 1).reshape(batch, 144, 32)
    return torch.cat([image_tokens, coords], -1)


def predict_pixel(model, x, context):
    batch = x.shape[0]
    with torch.autocast('mps', dtype=torch.bfloat16):
        condition = model.raster_conditioner(raster_tokens(context))
        result = model(hidden_states=x, encoder_hidden_states=condition,
                       encoder_attention_mask=torch.ones(batch, 144, device=x.device),
                       timestep=torch.full((batch,), 1000., device=x.device), return_dict=False)[0]
    return result.float()


def sample_pixel(model, contexts, seed, steps):
    assert steps == 1, 'This arm prospectively uses one pure-noise Euler step'
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros(1, 1, 16, 16, device='mps')
    mask[:, :, 9:, :] = 1
    was_training = model.training
    model.eval()
    outputs = []
    # MPS BF16 inference kernels differ materially from the trained autograd
    # path on this stack. Track the forward graph, detach immediately, and
    # perform no backward pass or parameter update during generation.
    with torch.enable_grad():
        for start in range(0, len(contexts), 4):
            c = contexts[start:start+4].to('mps')
            x = torch.randn(c.shape, generator=generator).to('mps')*mask + c*(1-mask)
            count = len(c)
            # The validated MPS path uses the training batch shape even for one image.
            if count < 4:
                x = torch.cat([x, x[-1:].repeat(4-count, 1, 1, 1)], dim=0)
                c = torch.cat([c, c[-1:].repeat(4-count, 1, 1, 1)], dim=0)
            outputs.append(((x-predict_pixel(model, x, c))*mask + c*(1-mask))[:count].detach().cpu())
    model.train(was_training)
    return torch.cat(outputs)


def main():
    from diffusers import AutoencoderDC
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--skip-development', action='store_true')
    args = parser.parse_args()
    root = args.root
    sys.path.insert(0, str(root / 'source'))
    from common import atomic_json, digest, pixels, render
    protocol = root / 'source/mac_sana_pixel_v1.json'
    config = json.loads(protocol.read_text())
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.65)
    torch.manual_seed(config['seed'])
    critic, critic_sha256 = None, None
    if 'critic' in config:
        from ocr_critic import DigitCritic
        from safetensors.torch import load_file
        critic_path = root / 'critic/model.safetensors'
        receipt = json.loads((root / 'critic/validation.json').read_text())
        critic_sha256 = digest(critic_path)
        assert receipt['passed'] and receipt['critic_sha256'] == critic_sha256
        assert receipt['protocol_sha256'] == config.get('critic_validation_protocol_sha256', digest(protocol))
        assert receipt['font_sha256'] == digest(root / 'fonts/train.ttf')
        if 'critic_sha256' in config:
            assert critic_sha256 == config['critic_sha256']
        critic = DigitCritic().eval().requires_grad_(False)
        critic.load_state_dict(load_file(str(critic_path), device='cpu'), strict=True)
        critic.to('mps')
    assert shutil.disk_usage(root).free / 2**30 > 41
    manifest = json.loads((root / 'data/manifest.json').read_text())
    rows = manifest['rows']
    cs, ys = [], []
    for row in rows:
        path = root / 'data' / row['file']
        assert digest(path) == row['sha256'], row['id']
        item = np.load(path)
        assert set(item.files) == {'context', 'target'}, row['id']
        cs.append(item['context'])
        ys.append(item['target'])
    contexts, targets = torch.tensor(np.stack(cs)), torch.tensor(np.stack(ys))
    answer_pixels = torch.tensor(np.stack([
        pixels(render(0, 0, 0, root / 'fonts/train.ttf', answer=i))[:, 288:, 128:384]
        for i in range(20)]))
    model, adapters = load_pixel_model(root)
    slow, fast = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (slow if any(name.startswith(f'transformer_blocks.{i}.ff.') for i in range(24, 28))
             else fast).append(parameter)
    parameters = slow + fast
    optimizer = torch.optim.AdamW([
        {'params': fast, 'lr': config['learning_rate']},
        {'params': slow, 'lr': config['final_feedforward_learning_rate']}],
        weight_decay=config['weight_decay'], foreach=False)
    output = root / ('pilot' if args.pilot else 'training')
    output.mkdir(exist_ok=True)
    resume = output / 'resume.pt'
    initial = root / 'initial.pt'
    if not resume.exists():
        receipt = root / 'initial-checkpoint.json'
        expected_initial = (json.loads(receipt.read_text())['sha256'] if receipt.exists()
                            else config['initial_checkpoint_sha256'])
        assert digest(initial) == expected_initial
    state = torch.load(resume if resume.exists() else initial, map_location='cpu', weights_only=False)
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(state['trainable']) <= expected
    if 'critic' in config or 'contrast' in config:
        assert set(state['trainable']) == expected
    assert not model.load_state_dict(state['trainable'], strict=False).unexpected_keys
    rng = np.random.default_rng(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    start = 0
    if resume.exists():
        assert set(state['trainable']) == expected
        assert state['pixel_protocol_sha256'] == digest(protocol)
        assert state['data_sha256'] == digest(root / 'data/manifest.json')
        if critic is not None:
            assert state['critic_sha256'] == critic_sha256
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['numpy_rng']
        generator.set_state(state['torch_rng'])
        start = state['step']
    assert start < args.end, (start, args.end)
    del state
    snapshot = Path(json.loads((root / 'model.json').read_text())['snapshot'])
    vae = AutoencoderDC.from_pretrained(snapshot, subfolder='vae', variant='fp16', torch_dtype=torch.float32)
    scale = vae.config.scaling_factor
    decoder = checkpoint_decoder(vae.decoder.eval().requires_grad_(False).to('mps'))
    del vae
    gc.collect()
    torch.mps.empty_cache()
    atomic_json(output / 'model-audit.json', {
        'trainable_parameters': sum(p.numel() for p in parameters),
        'trainable_names': sorted(expected), 'adapter_modules': adapters,
        'decoder_parameters': sum(p.numel() for p in decoder.parameters()),
        'pixel_protocol_sha256': digest(protocol),
        'data_sha256': digest(root / 'data/manifest.json'),
        'critic_sha256': critic_sha256})
    copy_ids = example_indices(rows, config, copy=True)
    add_ids = example_indices(rows, config, copy=False)
    assert copy_ids and add_ids
    mask = torch.zeros(1, 1, 16, 16, device='mps')
    mask[:, :, 9:, :] = 1
    model.train()
    times = []

    def save(step):
        state = {'step': step, 'protocol_sha256': digest(root / 'source/mac_sana_column_v1.json'),
                 'pixel_protocol_sha256': digest(protocol), 'data_sha256': digest(root / 'data/manifest.json'),
                 'trainable': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                 'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
                 'torch_rng': generator.get_state()}
        if critic is not None:
            state['critic_sha256'] = critic_sha256
        temporary = output / 'resume.tmp.pt'
        torch.save(state, temporary)
        temporary.replace(resume)
        atomic_json(output / 'checkpoint.json', {'step': step, 'sha256': digest(resume),
                                                'bytes': resume.stat().st_size})

    for step in range(start+1, args.end+1):
        tick = time.monotonic()
        move_optimizer_moments(optimizer, 'cpu')
        torch.mps.empty_cache()
        ids = [int(rng.choice(copy_ids if step <= config['copy_only_updates'] or
                              rng.random() < config['copy_replay_probability_after_curriculum']
                              else add_ids)) for _ in range(config['batch_size'])]
        c, y = contexts[ids].to('mps'), targets[ids].to('mps')
        noise = torch.randn(y.shape, generator=generator).to('mps')
        x = noise*mask + c*(1-mask)
        slot = (step-1) % len(ids)
        # Obtain the pixel-loss evaluation point in the same train/autograd mode
        # as the later recomputation, then release the SANA graph before VAE AD.
        velocity = predict_pixel(model, x, c)
        clean = (x-velocity)*mask + c*(1-mask)
        pixel_point = clean[slot:slot+1].detach().cpu()
        del velocity, clean
        # The two graphs are disjoint. Evict frozen and trainable SANA weights
        # during decoder differentiation, then restore the exact FP32 tensors.
        model.to('cpu')
        gc.collect()
        torch.mps.empty_cache()
        pixel_latents = pixel_point.to('mps').requires_grad_(True)
        target_pixels = answer_pixels[rows[ids[slot]]['answer']:rows[ids[slot]]['answer']+1].to('mps')
        with torch.autocast('mps', dtype=torch.bfloat16):
            decoded = decoder(pixel_latents / scale)[:, :, 288:, 128:384].float()
        foreground = 1 + 4*(1-target_pixels)/2
        pixel_loss = ((decoded-target_pixels).square()*foreground).sum() / foreground.sum()
        assert torch.isfinite(pixel_loss).item(), step
        semantic = semantic_loss(decoded, answer_pixels.to('mps'),
                                 torch.tensor([rows[ids[slot]]['answer']], device='mps'), config, critic)
        semantic_value = float(semantic.detach())
        (pixel_loss + semantic).backward()
        pixel_value = float(pixel_loss.detach())
        pixel_gradient = pixel_latents.grad.detach().cpu()
        assert torch.isfinite(pixel_gradient).all()
        assert all(p.grad is None for p in parameters), 'Pixel phase must be detached from SANA'
        if critic is not None:
            assert all(not p.requires_grad and p.grad is None for p in critic.parameters())
        del pixel_latents, target_pixels, decoded, foreground, pixel_loss, semantic
        gc.collect()
        torch.mps.empty_cache()
        model.to('mps')
        velocity = predict_pixel(model, x, c)
        clean = (x-velocity)*mask + c*(1-mask)
        delta = (clean[slot:slot+1].detach().cpu()-pixel_point).abs()
        equivalence = {'step': step, 'maximum_absolute_difference': float(delta.max()),
                       'mean_absolute_difference': float(delta.mean()), 'maximum_allowed': 1e-6}
        equivalent = equivalence['maximum_absolute_difference'] <= 1e-6
        if step == start+1 or not equivalent:
            atomic_json(output / f'vjp-point-{step:05d}.json', {**equivalence, 'passed': equivalent})
        assert equivalent, json.dumps(equivalence)
        weight = 1 + 4*(y-c).abs().mean(1, keepdim=True).clamp(0, 1)
        flow = ((clean-y).square()*mask*weight).sum() / (mask.expand_as(y)*weight).sum()
        # This scalar supplies exactly J_SANA^T times the detached pixel gradient;
        # report the actual pixel loss, not the surrogate's arbitrary scalar value.
        surrogate = pixel_vjp_surrogate(clean[slot:slot+1], pixel_gradient.to('mps'))
        loss = flow + surrogate
        assert torch.isfinite(loss).item(), step
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert math.isfinite(float(norm)), step
        torch.mps.empty_cache()
        move_optimizer_moments(optimizer, 'mps')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        elapsed = time.monotonic()-tick
        times.append(elapsed)
        record = {'step': step, 'example_ids': [rows[i]['id'] for i in ids],
                  'flow_loss': float(flow.detach()), 'pixel_loss': pixel_value,
                  'total_loss': float(flow.detach())+pixel_value+semantic_value,
                  'weighted_semantic_loss': semantic_value, 'gradient_norm': float(norm), 'seconds': elapsed,
                  'vjp_point_max_absolute_difference': equivalence['maximum_absolute_difference'],
                  'mps_gib': torch.mps.driver_allocated_memory()/2**30,
                  'phase': 'copy' if step <= config['copy_only_updates'] else 'add-with-copy-replay'}
        with (output / 'metrics.jsonl').open('a') as log:
            log.write(json.dumps(record)+'\n')
        del velocity, clean, flow, surrogate, loss
        if step % 10 == 0 or step == start+1:
            print(json.dumps(record), flush=True)
        if step % 100 == 0 and not args.pilot:
            save(step)
        if shutil.disk_usage(root).free / 2**30 < 40:
            if not args.pilot:
                save(step)
            raise RuntimeError('Disk reserve reached')
    if args.pilot:
        median = float(np.median(times[2:]))
        atomic_json(output / 'summary.json', {
            'updates': len(times), 'median_seconds': median,
            'mps_gib': torch.mps.driver_allocated_memory()/2**30,
            'passed': median <= config['resource_pilot']['median_update_seconds_maximum']})
    else:
        save(args.end)
        if args.skip_development:
            return
        del decoder, critic
        torch.mps.empty_cache()
        ids = example_indices(rows, config, development=True)
        generated = sample_pixel(model, contexts[ids], config.get('binary', {}).get('development_noise_seed', 1100910), 1)
        np.savez_compressed(output / f'development-{args.end:05d}.npz',
                            latents=generated.numpy(), indices=np.array(ids))
        print('PIXEL_SEGMENT_COMPLETE', args.end, flush=True)


if __name__ == '__main__':
    main()
