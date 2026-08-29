"""Visualization helpers for MIBiG embeddings."""


def save_bgc_class_umap(*args, **kwargs):
    from .umap_plot import save_bgc_class_umap as _save_bgc_class_umap

    return _save_bgc_class_umap(*args, **kwargs)


def save_joint_umap(*args, **kwargs):
    from .umap_plot import save_joint_umap as _save_joint_umap

    return _save_joint_umap(*args, **kwargs)

__all__ = ["save_bgc_class_umap", "save_joint_umap"]
