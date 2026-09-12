"""Stage the same test suite without making the source package importable."""
from pathlib import Path
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys


def main():
    source = Path(__file__).resolve().parents[1]
    target = Path(sys.argv[1]).resolve()
    version = os.environ.get("VASCX_VERSION", "").strip()
    requirement = "retinalysis-vascx" + (f"=={version}" if version else "")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", requirement], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    for distribution, module in [
        ("retinalysis-vascx", "vascx"),
        ("retinalysis-inference", "rtnls_inference"),
        ("retinalysis-enface", "rtnls_enface"),
    ]:
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            raise SystemExit(
                f"Published {requirement} did not install required dependency {distribution}"
            ) from None
        location = Path(importlib.import_module(module).__file__).resolve()
        assert Path(sys.prefix).resolve() in location.parents, f"Non-isolated import: {location}"
        print(f"{distribution}=={installed_version}: {location}", flush=True)
    for name in ("tests", "samples"):
        shutil.copytree(source / name, target / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copy2(source / "pyproject.toml", target / "pyproject.toml")


if __name__ == "__main__":
    main()
