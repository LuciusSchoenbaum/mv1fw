

from matplotlib.font_manager import fontManager, FontProperties
from matplotlib import rc



# helper
def font_name2filename(name):
    """
    Return the filename storing the font from a requested font name.

    Parameters:

        name (string):

    """
    return fontManager.findfont(FontProperties(family=name))


# helper
def set_rc_serif():
    """
    Find a serif font, and use Times if available.
    If Times is not found, the generic name "serif" will (probably?)
    set matplotlib's font to DejaVuSerif.

    """
    names = fontManager.get_font_names()
    if "Times" in names:
        name = "Times"
    else:
        name = "serif"
    rc('font',**{'family':name})




