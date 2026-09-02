"""Run manifest: everything needed to reproduce a training run."""
import datetime
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, sha256_file  # noqa: E402


def collect_env():
    import accelerate
    import peft
    import torch
    import transformers
    env = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'peft': peft.__version__,
        'accelerate': accelerate.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env['cuda'] = torch.version.cuda
        env['device'] = torch.cuda.get_device_name(0)
        env['capability'] = '.'.join(
            map(str, torch.cuda.get_device_capability(0)))
    return env


def git_commit(repo_dir):
    try:
        head = subprocess.run(
            ['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ['git', '-C', str(repo_dir), 'status', '--porcelain'],
            capture_output=True, text=True, check=True).stdout.strip()
        return head + ('+dirty' if dirty else '')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def build_manifest(args, files, extra):
    git = {'segue': git_commit(REPO_ROOT)}
    minuspod = REPO_ROOT.parent / 'MinusPod'
    if minuspod.is_dir():
        git['minuspod'] = git_commit(minuspod)
    m = {
        'created_at': datetime.datetime.now(datetime.timezone.utc)
                      .strftime('%Y-%m-%dT%H:%M:%SZ'),
        'args': dict(args),
        'sha256': {name: sha256_file(p) for name, p in files.items()},
        'git': git,
    }
    m.update(extra)
    return m
