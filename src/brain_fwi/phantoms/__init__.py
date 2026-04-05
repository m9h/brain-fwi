"""Head phantom loading and acoustic property mapping."""

from .brainweb import load_brainweb_phantom, load_brainweb_slice, make_synthetic_head
from .properties import (
    TISSUE_PROPERTIES,
    map_labels_to_speed,
    map_labels_to_density,
    map_labels_to_attenuation,
    map_labels_to_all,
)
