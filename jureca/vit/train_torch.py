"""Simplest possible ViT training on CIFAR-100, in plain PyTorch.

This is the reference: the Tenstorrent version must reproduce its loss curve.
Model and recipe follow ViT_Baseline_config.pdf (patch size 4, 64 tokens, no CLS token,
mean pooling, AdamW, warmup + cosine, label smoothing 0.1, batch size 512), without augmentation.
Where every choice comes from: README.md and test/REPORT.md.

Files: common.py (data, batch order, settings, schedule, logging: shared with the Tenstorrent version),
       model_torch.py (the model), this file (PyTorch optimizer and training loop).

Run:  python train_torch.py --model tiny --epochs 200
"""

import os
import time

import torch
import torch.nn.functional as F

import common
from model_torch import ViT


@torch.no_grad()
def evaluate(model, images, labels, batch_size):
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, len(images), batch_size):
        logits = model(images[i : i + batch_size])
        y = labels[i : i + batch_size]
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
    model.train()
    return loss_sum / len(images), correct / len(images)


def make_optimizer(model, lr, weight_decay):
    """AdamW with weight decay on the weight matrices only, not on biases, LayerNorm parameters or the position table."""
    no_decay = [p for name, p in model.named_parameters() if p.dim() < 2 or name == "pos_embed"]
    decay = [p for name, p in model.named_parameters() if p.dim() >= 2 and name != "pos_embed"]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def main():
    args = common.parse_args(default_out="runs/torch")

    torch.manual_seed(args.seed)  # for the model initialisation
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))  # numb slurm cpus or default 4
    log = common.Log(args.out)

    train_x, train_y = common.load_cifar100(args.data, "train")
    test_x, test_y = common.load_cifar100(args.data, "test")
    if args.images:
        train_x, train_y = train_x[: args.images], train_y[: args.images]
        test_x, test_y = test_x[: args.images], test_y[: args.images]

    model = ViT(**common.MODELS[args.model])
    print(f"ViT-{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    # Starting weights, so that another implementation can start from exactly the same numbers.
    torch.save(model.state_dict(), os.path.join(args.out, "init.pt"))

    peak_lr = args.lr * args.batch_size / 512  # DeiT main.py L350
    optimizer = make_optimizer(model, peak_lr, args.weight_decay)

    # Own random stream for the batch order: it depends only on the seed, not on the model's random draws.
    data_order = torch.Generator().manual_seed(args.seed)

    step = 0
    for epoch in range(args.epochs):
        lr = common.lr_at(epoch, args.epochs, args.warmup_epochs, peak_lr)  # one learning rate per epoch
        for group in optimizer.param_groups:
            group["lr"] = lr
        t0, loss_sum = time.time(), 0.0
        for i, idx in enumerate(common.batches(len(train_x), args.batch_size, data_order)):
            x, y = train_x[idx], train_y[idx]

            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            step += 1
            log.step(step, loss.item(), lr)
            if args.max_steps and step >= args.max_steps:
                break

        images_per_s = (i + 1) * args.batch_size / (time.time() - t0)
        test_loss, test_acc = evaluate(model, test_x, test_y, args.batch_size)
        log.epoch(epoch + 1, step, loss_sum / (i + 1), test_loss, test_acc, images_per_s)
        if args.max_steps and step >= args.max_steps:
            break

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()
