from scripts._bootstrap import ensure_src_path

ensure_src_path()

from scripts.train_contrastive import main


if __name__ == "__main__":
    main()
