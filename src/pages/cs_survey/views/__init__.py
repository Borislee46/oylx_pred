from .base import VIEW_REGISTRY, register_view
from .matrix_by_pillar import render as render_matrix_by_pillar
from .matrix_by_product import render as render_matrix_by_product
from .overview import render as render_overview

register_view("overview", render_overview)
register_view("matrix_by_product", render_matrix_by_product)
register_view("matrix_by_pillar", render_matrix_by_pillar)

__all__ = [
    "VIEW_REGISTRY",
    "register_view",
    "render_overview",
    "render_matrix_by_product",
    "render_matrix_by_pillar",
]
