import json
import re
import shutil
from pathlib import Path

source_root = Path("third_party/MAIC/refactor")
copy_root = Path("runs_diagnostic/source_exactkl")
run_dir = Path("runs_diagnostic/PushBox16/MAIC/seed43_exactKL")

if copy_root.exists() or run_dir.exists():
    raise SystemExit("Destination exists; refusing to overwrite.")

source_file = source_root / "algorithms/MAIC.py"
code = source_file.read_text()

def replace_once(pattern, replacement):
    global code
    code, count = re.subn(
        pattern, replacement, code, count=1, flags=re.M
    )
    if count != 1:
        raise RuntimeError(f"Expected one match: {pattern}")

# Remove only the replay-based KL surrogate.
replace_once(
    r'\(alpha1\s*\*\s*self\.scaled_entropy_coef\s*\+\s*self\.alpha_kl\)'
    r'\s*\*\s*\(joint_new_log_probs\s*\+\s*1\)'
    r'\s*-\s*self\.alpha_kl\s*\*\s*joint_old_log_probs',
    '(alpha1 * self.scaled_entropy_coef) * (joint_new_log_probs + 1)'
)

# Accumulate the exact KL calculated for every agent.
replace_once(
    r'^([ \t]*)total_policy_loss\s*=\s*0\.0[ \t]*$',
    r'\g<0>\n\g<1>total_exact_kl = 0.0'
)

replace_once(
    r'^([ \t]*)total_policy_loss\s*\+=\s*policy_loss_a[ \t]*$',
    r'\g<0>\n\g<1>total_exact_kl += kl_divergence'
)

# Add exact KL to the loss before backpropagation.
replace_once(
    r'^([ \t]*)self\.actor_optimizer\.zero_grad\(\)[ \t]*$',
    r'\g<1>total_policy_loss = total_policy_loss + self.alpha_kl * total_exact_kl\n\g<0>'
)

shutil.copytree(
    source_root,
    copy_root,
    ignore=shutil.ignore_patterns(
        "__pycache__", ".git", "*.pth", "*.mp4"
    )
)

(copy_root / "algorithms/MAIC.py").write_text(code)

cfg = json.loads(Path(
    "runs_diagnostic/PushBox16/MAIC/seed43/config.json"
).read_text())

assert cfg["seed"] == 43
assert cfg["scaled_information_gain_coef"] == 1
assert cfg["scaled_entropy_coef"] == 1

cfg["model_config"]["alpha_kl"] = 1.0
cfg["model_config"]["policy_update_steps"] = 3
cfg["episodes"] = 60
cfg["save_every"] = 1
cfg["metrics_save_every"] = 1
cfg["model_dir"] = str(run_dir.resolve())

run_dir.mkdir(parents=True)
(run_dir / "config.json").write_text(
    json.dumps(cfg, indent=2)
)

print("Created exact-KL source:", copy_root)
print("Created D03 configuration:", run_dir)
