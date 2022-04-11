import typer
from pathlib import Path

from pip._vendor.distlib.compat import raw_input
from tqdm import tqdm


def main(
    input_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    )
):
    print(f"Found {len(list(input_dir.rglob('*.pkl')))} models in {input_dir}")
    delete = raw_input('Would you like to delete all models (y/n)? ')

    if delete.lower() == 'yes' or delete.lower() == 'y':
        print("deleting...")
        for p in tqdm(input_dir.rglob("*.pkl")):
            p.unlink()
        print("done.")
    else:
        print("cancel.")


if __name__ == "__main__":
    typer.run(main)
