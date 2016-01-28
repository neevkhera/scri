# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

"""Submodule for operating on SpEC waveform and metadata files"""

from .._version import __version__

from .metadata import (read_metadata, read_metadata_into_object,)
from .com_motion import (com_motion, estimate_avg_com_motion, remove_avg_com_motion)
from .file_io import (read_from_h5, write_to_h5,)
