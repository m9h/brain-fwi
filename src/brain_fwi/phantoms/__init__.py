"""Head phantom loading and acoustic property mapping."""

from .brainweb import load_brainweb_phantom, load_brainweb_slice, make_synthetic_head
from .synthetic import make_three_layer_head
from .itrusst import (
    make_bm1_water_box,
    make_bm2_single_layer_plate,
    make_bm3_three_layer_plate,
    make_bm4_curved_three_layer_plate,
)
from .tfuscapes import (
    ct_to_acoustic,
    discover_tfuscapes_samples,
    head_mask_from_ct,
    load_tfuscapes_sample,
)
from .sci_head import (
    SCI_ACOUSTIC_PROPERTIES,
    SCI_LABEL_NAMES,
    load_sci_head_acoustic,
    load_sci_head_segmentation,
    map_sci_labels_to_acoustic,
)
from .simnibs import (
    SIMNIBS_LABEL_NAMES,
    find_simnibs_tissue_labeling,
    load_simnibs_acoustic,
    load_simnibs_segmentation,
    map_simnibs_labels_to_acoustic,
    simnibs_acoustic_table,
)
from .properties import (
    TISSUE_PROPERTIES,
    map_labels_to_speed,
    map_labels_to_density,
    map_labels_to_attenuation,
    map_labels_to_all,
)
from .mida import (
    MIDA_TISSUE_GROUPS,
    MIDA_ACOUSTIC_PROPERTIES,
    map_mida_labels_to_acoustic,
    load_mida_volume,
    load_mida_acoustic,
    resample_volume,
)
