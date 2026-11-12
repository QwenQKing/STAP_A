import sys
from pathlib import Path
import importlib.util

target = Path(__file__).with_name("evaluate_w0-w4.py")
spec = importlib.util.spec_from_file_location("evaluate_w0_w4", target)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.argv[1:] = ["--metrics", "clip_hps"] + sys.argv[1:]
mod.main()

