"""Is our baseline the same as the reference implementations?

Ours:      ../common.py, ../model_torch.py, ../train_torch.py
Reference: timm's VisionTransformer, trained the way DeiT's main.py does it (timm's optimizer factory,
           cosine scheduler and label-smoothing loss). The DeiT code is in reference/ for reading.
Nothing on the reference side comes from our code, except the loaded images and the order of the batches.

Run from the repository root:  python jureca/vit/test/test_against_reference.py     (about 3 minutes)
Results and explanations:      REPORT.md
"""

import argparse
import csv
import os
import sys
import tempfile
from functools import partial

import torch
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy
from timm.models.vision_transformer import VisionTransformer
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
import train_torch
from model_torch import ViT

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))


def timm_vit(depth, dim, heads, mlp_dim):
    """The reference model. Second line: as in reference/deit/models.py L65-66. Third line: our two
    deviations (patch size 4 on 32x32 images; mean pooling after the final norm instead of a class token)."""
    return VisionTransformer(
        embed_dim=dim, depth=depth, num_heads=heads, mlp_ratio=mlp_dim / dim, num_classes=100,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_size=32, patch_size=4, class_token=False, global_pool="avg", fc_norm=False,
    )  # fmt: skip


def timm_name(name):
    """Our parameter names in timm's spelling: the MLP layers are fc1/fc2, the patch embedding is .proj."""
    name = name.replace(".mlp.0.", ".mlp.fc1.").replace(".mlp.2.", ".mlp.fc2.")
    return name.replace("patch_embed.", "patch_embed.proj.")


def to_timm(weights):
    """Our weights under timm's names. timm's patch embedding is a convolution (D, 3, 4, 4), ours a
    linear layer (D, 48), where the 48 inputs of a patch are ordered (row, column, colour)."""
    out = {timm_name(name): w for name, w in weights.items()}
    out["patch_embed.proj.weight"] = out["patch_embed.proj.weight"].view(-1, 4, 4, 3).permute(0, 3, 1, 2)
    return out


def deit_args(epochs, warmup_epochs, batch_size):
    """The defaults of reference/deit/main.py (L56-92) that the optimizer and scheduler factories read."""
    return argparse.Namespace(
        opt="adamw", opt_eps=1e-8, opt_betas=None, momentum=0.9, weight_decay=0.05,
        lr=5e-4 * batch_size / 512,  # main.py L350
        sched="cosine", epochs=epochs, warmup_epochs=warmup_epochs, warmup_lr=1e-6, min_lr=1e-5,
        decay_epochs=30, decay_rate=0.1, cooldown_epochs=10, patience_epochs=10,
        lr_noise=None, lr_noise_pct=0.67, lr_noise_std=1.0, seed=0,
    )  # fmt: skip


def check(name, difference, tolerance):
    ok = difference <= tolerance
    print(f"{'ok  ' if ok else 'FAIL'}  {name}: difference {difference:.1e} (allowed {tolerance:.0e})")
    return ok


def test_normalisation():
    ours = torch.cat([common.MEAN.flatten(), common.STD.flatten()])
    reference = torch.tensor(IMAGENET_DEFAULT_MEAN + IMAGENET_DEFAULT_STD)  # reference/deit/datasets.py L108
    return check("normalisation constants", (ours - reference).abs().max().item(), 0)


def test_model(size, images, labels):
    """Same weights and same images must give the same class scores and the same gradients."""
    torch.manual_seed(0)
    ours = ViT(**common.MODELS[size])
    # At initialisation all biases are 0 and all LayerNorm weights 1, which would hide mistakes in them.
    for p in ours.parameters():
        p.data += 0.05 * torch.randn_like(p)
    reference = timm_vit(**common.MODELS[size])
    reference.load_state_dict(to_timm(ours.state_dict()))  # fails if names or shapes differ

    out_ours, out_reference = ours(images), reference(images)
    nn.functional.cross_entropy(out_ours, labels).backward()
    nn.functional.cross_entropy(out_reference, labels).backward()
    grads_ours = to_timm({name: p.grad for name, p in ours.named_parameters()})
    grad_difference = max(
        ((grads_ours[name] - p.grad).abs().max() / p.grad.abs().max()).item()
        for name, p in reference.named_parameters()
    )
    scale = out_reference.abs().max().item()
    a = check(f"model {size}: class scores", (out_ours - out_reference).abs().max().item() / scale, 1e-5)
    b = check(f"model {size}: gradients", grad_difference, 1e-4)
    return a and b


def test_schedule(epochs, warmup_epochs, batch_size):
    """Our learning rate of every epoch against timm's scheduler, created as in reference/deit/main.py L355."""
    args = deit_args(epochs, warmup_epochs, batch_size)
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=args.lr)
    scheduler, _ = create_scheduler(args, optimizer)
    worst = 0.0
    for epoch in range(epochs):
        scheduler.step(epoch)
        ours, reference = common.lr_at(epoch, epochs, warmup_epochs, args.lr), optimizer.param_groups[0]["lr"]
        worst = max(worst, abs(ours - reference) / reference)
    return check(f"schedule ({epochs} epochs, {warmup_epochs} warmup, batch size {batch_size})", worst, 1e-12)


def test_optimizer(defaults):
    """Our optimizer against timm's factory, called as in reference/deit/main.py L352: every single
    parameter must get the same weight decay, learning rate and Adam settings."""
    ours_model, reference_model = ViT(**common.MODELS["tiny"]), timm_vit(**common.MODELS["tiny"])
    ours = train_torch.make_optimizer(ours_model, defaults.lr, defaults.weight_decay)
    reference = create_optimizer(deit_args(200, 5, 512), reference_model)

    def settings(model, optimizer, rename):
        group = {id(p): g for g in optimizer.param_groups for p in g["params"]}
        keys = ["weight_decay", "lr", "betas", "eps"]
        return {rename(name): [group[id(p)][k] for k in keys] for name, p in model.named_parameters()}

    a, b = settings(ours_model, ours, timm_name), settings(reference_model, reference, str)
    different = sorted(name for name in a.keys() | b.keys() if a.get(name) != b.get(name))
    for name in different[:5]:  # the first few are enough to see what is wrong
        print(f"      {name}: ours {a.get(name)}, reference {b.get(name)}")
    return check(f"optimizer: settings of the {len(b)} parameters", len(different), 0)


def test_training(data, epochs=4, warmup_epochs=2, batch_size=256, images=1024):
    """Run our real training script on a few images, then repeat the run with the reference pieces,
    starting from the weights our script saved. Learning rates, losses and final weights must agree."""
    out = tempfile.mkdtemp()
    sys.argv = ["train_torch.py", "--model", "tiny", "--epochs", str(epochs), "--warmup-epochs", str(warmup_epochs),
                "--batch-size", str(batch_size), "--images", str(images), "--data", data, "--out", out]  # fmt: skip
    train_torch.main()
    with open(os.path.join(out, "steps.csv")) as f:
        ours = [(float(row["loss"]), float(row["lr"])) for row in csv.DictReader(f)]
    ours_final = to_timm(torch.load(os.path.join(out, "model.pt")))

    args = deit_args(epochs, warmup_epochs, batch_size)
    model = timm_vit(**common.MODELS["tiny"])
    model.load_state_dict(to_timm(torch.load(os.path.join(out, "init.pt"))))
    optimizer = create_optimizer(args, model)  # reference/deit/main.py L352
    scheduler, _ = create_scheduler(args, optimizer)  # L355
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)  # L363

    train_x, train_y = common.load_cifar100(data, "train")
    train_x, train_y = train_x[:images], train_y[:images]
    data_order = torch.Generator().manual_seed(42)  # the default seed of our script
    reference = []
    for epoch in range(epochs):
        scheduler.step(epoch)  # DeiT calls this after the epoch (L432); see REPORT.md, "One deliberate difference"
        for idx in common.batches(len(train_x), batch_size, data_order):
            loss = criterion(model(train_x[idx]), train_y[idx])  # reference/deit/engine.py L46-68
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reference.append((loss.item(), optimizer.param_groups[0]["lr"]))

    print("  step   learning rate   loss ours   loss reference")
    for step, ((loss_a, lr_a), (loss_b, lr_b)) in enumerate(zip(ours, reference), 1):
        print(f"  {step:4d}   {lr_a:.3e}       {loss_a:.6f}    {loss_b:.6f}")
    loss_difference = max(abs(a[0] - b[0]) for a, b in zip(ours, reference))
    lr_difference = max(abs(a[1] - b[1]) / b[1] for a, b in zip(ours, reference))
    weight_difference = max((ours_final[name] - w).abs().max().item() for name, w in model.state_dict().items())
    a = check("training: number of steps", abs(len(ours) - len(reference)), 0)
    b = check("training: learning rate of every step", lr_difference, 1e-6)
    c = check("training: loss of every step", loss_difference, 1e-5)
    d = check("training: final weights", weight_difference, 1e-5)
    return a and b and c and d


if __name__ == "__main__":
    defaults = common.parse_args(default_out="")
    data = defaults.data
    images, labels = common.load_cifar100(data, "train")
    results = [test_normalisation()]
    results += [test_model(size, images[:16], labels[:16]) for size in common.MODELS]
    results += [test_schedule(200, 5, 512), test_schedule(300, 5, 1024), test_schedule(4, 2, 256)]
    results += [test_optimizer(defaults)]
    results += [test_training(data)]
    print("ALL OK" if all(results) else "SOMETHING FAILED")
