"""Export a self-contained Hugging Face Docker Space from the research repository.

    python scripts/package_space.py --output /tmp/EyeQC_space

The output is the *contents* to push to huggingface.co/spaces/Dharini24/EyeQC.
It intentionally omits participant-level research records and larger development caches.
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES=('docs','portal','eyeqc','eyeqc_runtime','eyeqc_flair','eyeqc_vessel','models')
FILES=('requirements.txt','requirements-optional.txt')
IGNORE=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc','*.pyo','.DS_Store')

def export(destination:Path)->None:
    destination=destination.resolve()
    if destination==ROOT or ROOT in destination.parents:raise ValueError('Choose an output directory outside the research repository')
    destination.mkdir(parents=True,exist_ok=True)
    for directory in DIRECTORIES:
        shutil.copytree(ROOT/directory,destination/directory,dirs_exist_ok=True,ignore=IGNORE)
    for filename in FILES:shutil.copy2(ROOT/filename,destination/filename)
    for filename in ('Dockerfile','README.md','.gitignore'):shutil.copy2(ROOT/'space'/filename,destination/filename)
    # Ensure the dedicated Space bundle can run without the separately held research files.
    assert (destination/'portal/app.py').is_file()
    assert (destination/'docs/index.html').is_file()
    assert (destination/'docs/assets/data/gallery.json').is_file()
    print(f'Hugging Face Space ready: {destination}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    export(Path(args.output))
