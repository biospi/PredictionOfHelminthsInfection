from pathlib import Path

if __name__ == "__main__":
    results = []
    folders = [x for x in Path("E:/Cats/article/ml_build_permutations_qnf_final/").glob('*/*/*') if x.is_dir()]
    for i, item in enumerate(folders):
        print(f"{i}/{len(folders)}...")