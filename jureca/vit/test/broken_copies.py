"""Does the test notice mistakes?

Makes copies of our code with one deliberate mistake each and runs test_against_reference.py on every
copy. A good test fails on every single one. Our real files are not touched.

Run from the repository root:  python jureca/vit/test/broken_copies.py     (about 10 minutes, 20 CPU cores)
"""

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

TEST = os.path.dirname(os.path.abspath(__file__))
VIT = os.path.dirname(TEST)

# What is wrong: (file, [(correct text, wrong text), ...])
MISTAKES = {
    "LayerNorm epsilon 1e-5 instead of 1e-6": ("model_torch.py", [("eps=1e-6", "eps=1e-5")]),
    "tanh approximation of GELU": ("model_torch.py", [("nn.GELU()", 'nn.GELU(approximate="tanh")')]),
    "tokens averaged before the final LayerNorm instead of after": (
        "model_torch.py",
        [
            (
                "        x = self.norm(x)\n        x = x.mean(dim=1)",
                "        x = x.mean(dim=1)\n        x = self.norm(x)",
            )
        ],
    ),
    "attention scores divided by sqrt(D) instead of sqrt(head dimension)": (
        "model_torch.py",
        [("math.sqrt(self.head_dim)", "math.sqrt(D)")],
    ),
    "pixels inside a patch flattened in a different order": (
        "model_torch.py",
        [("x.permute(0, 2, 4, 3, 5, 1)", "x.permute(0, 2, 4, 1, 3, 5)")],
    ),
    "no bias on q, k, v": (
        "model_torch.py",
        [
            ("nn.Linear(dim, 3 * dim)", "nn.Linear(dim, 3 * dim, bias=False)"),
            (
                "                nn.init.zeros_(m.bias)",
                "                if m.bias is not None:\n                    nn.init.zeros_(m.bias)",
            ),
        ],
    ),
    "weight decay also on biases, LayerNorm and position embedding": (
        "train_torch.py",
        [('{"params": no_decay, "weight_decay": 0.0}', '{"params": no_decay, "weight_decay": weight_decay}')],
    ),
    "weight decay also on the position embedding": (
        "train_torch.py",
        [(' or name == "pos_embed"', ""), (' and name != "pos_embed"', "")],
    ),
    "weight decay 0.04 instead of 0.05": (
        "common.py",
        [('"--weight-decay", type=float, default=0.05', '"--weight-decay", type=float, default=0.04')],
    ),
    "Adam epsilon 1e-6 instead of 1e-8": ("train_torch.py", [("eps=1e-8", "eps=1e-6")]),
    "Adam beta2 0.99 instead of 0.999": ("train_torch.py", [("betas=(0.9, 0.999)", "betas=(0.9, 0.99)")]),
    "no label smoothing": ("train_torch.py", [("label_smoothing=0.1", "label_smoothing=0.0")]),
    "label smoothing as worded in the DeiT paper (0.9 and 0.1/99)": (
        "train_torch.py",
        [
            (
                "loss = F.cross_entropy(logits, y, label_smoothing=0.1)",
                "loss = F.cross_entropy(logits, torch.full_like(logits, 0.1 / 99).scatter_(1, y[:, None], 0.9))",
            )
        ],
    ),
    "learning rate not scaled with the batch size": (
        "train_torch.py",
        [("peak_lr = args.lr * args.batch_size / 512", "peak_lr = args.lr")],
    ),
    "learning rate one epoch late (what DeiT's main.py literally does)": (
        "train_torch.py",
        [("common.lr_at(epoch, args.epochs", "common.lr_at(max(epoch - 1, 0), args.epochs")],
    ),
    "warmup starts at 0 instead of 1e-6": ("common.py", [("WARMUP_START_LR = 1e-6", "WARMUP_START_LR = 0.0")]),
    "cosine decays to 0 instead of 1e-5": ("common.py", [("MIN_LR = 1e-5", "MIN_LR = 0.0")]),
    "cosine counted from the end of warmup instead of from epoch 0": (
        "common.py",
        [
            (
                "math.cos(math.pi * epoch / epochs)",
                "math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs))",
            )
        ],
    ),
    "CIFAR mean instead of the ImageNet mean": (
        "common.py",
        [("MEAN = torch.tensor([0.485, 0.456, 0.406])", "MEAN = torch.tensor([0.5071, 0.4865, 0.4409])")],
    ),
}


def run(number, mistake, work):
    """Copy our code, put in one mistake, run the test on the copy. Returns which checks noticed it."""
    file, replacements = MISTAKES[mistake]
    folder = os.path.join(work, f"copy{number:02d}")
    os.makedirs(os.path.join(folder, "test"))
    for name in ["common.py", "model_torch.py", "train_torch.py"]:
        shutil.copy(os.path.join(VIT, name), folder)
    shutil.copy(os.path.join(TEST, "test_against_reference.py"), os.path.join(folder, "test"))

    path = os.path.join(folder, file)
    text = open(path).read()
    for correct, wrong in replacements:
        assert correct in text, f"'{correct}' not found in {file}: this script needs updating"
        text = text.replace(correct, wrong)
    open(path, "w").write(text)

    result = subprocess.run(
        [sys.executable, os.path.join("test", "test_against_reference.py")],
        cwd=folder, capture_output=True, text=True, env={**os.environ, "SLURM_CPUS_PER_TASK": "2"},
    )  # fmt: skip
    if "ALL OK" in result.stdout:
        return "NOT NOTICED"
    failed = {line[6:].split(":")[0] for line in result.stdout.splitlines() if line.startswith("FAIL")}
    if failed:
        return "noticed by: " + ", ".join(sorted(failed))
    return "noticed: the test stopped with an error (" + result.stderr.strip().splitlines()[-1][:60] + " ...)"


if __name__ == "__main__":
    work = tempfile.mkdtemp()
    with ThreadPoolExecutor(max_workers=10) as pool:
        verdicts = list(pool.map(lambda item: run(item[0], item[1], work), enumerate(MISTAKES, 1)))
    for mistake, verdict in zip(MISTAKES, verdicts):
        print(f"{mistake}\n    {verdict}")
    noticed = sum(verdict != "NOT NOTICED" for verdict in verdicts)
    print(f"\n{noticed} of {len(MISTAKES)} mistakes noticed")
    shutil.rmtree(work)
