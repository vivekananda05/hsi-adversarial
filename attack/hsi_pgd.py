"""PGD baseline from physics_spectral_attack.pdf, Eq. (12).

One CHW scene perturbation is shared by every overlapping patch. Budgets are
specified in 1/255 units and converted once inside attack_cube_pgd.
"""
from contextlib import contextmanager

import torch
from torch.nn import functional as F



def patchify_cube(cube, locations, patch_size):
    """Differentiable indexed slicing, with constant-zero padding at scene edges."""
    if cube.ndim != 3 or patch_size < 1 or patch_size % 2 != 1:
        raise ValueError('Expected a CHW cube and a positive odd patch size')
    locations = locations.to(device=cube.device, dtype=torch.long)
    height, width = cube.shape[-2:]
    if locations.ndim != 2 or locations.shape[1] != 2:
        raise ValueError('Locations must have shape [N, 2]')
    if ((locations < 0).any() or (locations[:, 0] >= height).any()
            or (locations[:, 1] >= width).any()):
        raise ValueError('Patch centers must lie inside the scene')
    offset = torch.arange(patch_size, device=cube.device) - patch_size // 2
    rows = locations[:, 0, None, None] + offset[None, :, None]
    cols = locations[:, 1, None, None] + offset[None, None, :]
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    patches = cube[:, rows.clamp(0, height - 1), cols.clamp(0, width - 1)]
    return patches.permute(1, 0, 2, 3) * valid[:, None].to(cube.dtype)


@contextmanager
def attack_mode(model):
    """Freeze BN statistics and defender weights while retaining input gradients."""
    modes = [(module, module.training) for module in model.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    model.eval()
    for parameter, _ in flags:
        parameter.requires_grad_(False)
    try:
        with torch.enable_grad():
            yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


def attack_cube_pgd(model, cube, locations, patch_size=13, epsilon=1,
                    alpha=1, steps=10, batch_size=16, norm='l_inf',
                    lower=0.0, upper=1.0, train_trades=False, labels=None):
    """Maximize cross-entropy, or TRADES KL when train_trades=True, on one cube.

    Evaluation must supply ALL evaluation centers in one call. Microbatches
    accumulate gradients before each projection, rather than attacking each
    patch batch independently. Training supplies only the current train centers.
    FAP supplies a whole scene already min-max scaled to [0,1]. Scaling is
    performed once BEFORE PGD, never recomputed during attack iterations.
    The default bounds keep the attacked cube in [0,1].
    Cross-entropy requires labels. TRADES uses detached clean predictions and
    needs a random start: KL has zero gradient at the exact clean input.
    """
    # Config/script value 1 means an effective input-cube budget of 1/255.
    epsilon = float(epsilon) / 255.0
    alpha = float(alpha) / 255.0
    if epsilon < 0 or alpha < 0 or steps < 0 or batch_size < 1:
        raise ValueError('Invalid PGD budget, step count, or microbatch size')
    if norm not in ('l_inf', 'l_2'):
        raise ValueError('Unsupported cube norm')
    if not torch.isfinite(cube).all():
        raise ValueError('Invalid cube values')
    lower = cube.min().item() if lower is None else lower
    upper = cube.max().item() if upper is None else upper
    if lower > upper:
        raise ValueError('Invalid cube clip bounds')
    if (cube < lower).any() or (cube > upper).any():
        raise ValueError('Clean cube is outside configured attack bounds')
    if len(locations) == 0:
        raise ValueError('Expected nonempty patch locations')
    if not train_trades:
        if labels is None or len(labels) != len(locations):
            raise ValueError('labels matching locations are required when train_trades=False')
        labels = labels.to(device=cube.device, dtype=torch.long)
    cube = cube.detach()
    locations = locations.to(cube.device)

    def project(delta):
        if norm == 'l_inf':
            delta = delta.clamp(-epsilon, epsilon)
        else:
            delta = delta * (epsilon / delta.norm().clamp_min(1e-12)).clamp(max=1)
        return (cube + delta).clamp(lower, upper) - cube

    delta = torch.zeros_like(cube)
    # Always use random initialization, matching the original RGB PGD.
    if epsilon > 0 and steps > 0:
        if norm == 'l_inf':
            delta.uniform_(-epsilon, epsilon)
        else:
            delta.normal_()
            delta *= epsilon * torch.rand((), device=cube.device) / delta.norm().clamp_min(1e-12)
        delta = project(delta)
    if steps == 0 or epsilon == 0:
        return torch.zeros_like(cube)
    with attack_mode(model):
        if train_trades:
            # TRADES target is detached and fixed throughout the inner maximization.
            with torch.no_grad():
                references = torch.cat([
                    model(patchify_cube(cube, locations[start:start + batch_size], patch_size)).softmax(dim=1)
                    for start in range(0, len(locations), batch_size)
                ])
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            grad = torch.zeros_like(delta)
            for start in range(0, len(locations), batch_size):
                pos = locations[start:start + batch_size]
                patches = patchify_cube(cube + delta, pos, patch_size)
                logits = model(patches)
                if train_trades:
                    objective = F.kl_div(
                        F.log_softmax(logits, dim=1),
                        references[start:start + batch_size],
                        reduction='sum') / len(locations)
                else:
                    objective = F.cross_entropy(
                        logits, labels[start:start + batch_size],
                        reduction='sum') / len(locations)
                grad.add_(torch.autograd.grad(objective, delta, only_inputs=True)[0])
            with torch.no_grad():
                direction = grad.sign() if norm == 'l_inf' else grad / grad.norm().clamp_min(1e-12)
                delta = project(delta + alpha * direction)
    return delta.detach()
