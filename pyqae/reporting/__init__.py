# This line will hide code by default when the notebook is exported as HTML
import os
import warnings

try:
    import IPython.core.display as di
except:
    class _junk(object):
        @staticmethod
        def display_html(in_html, *kwargs):
            # no output since we don't want the jquery code showing up
            pass

EXE_MODE = bool(os.environ.get('EXE_MODE', "False"))  # autoload when the python instance starts


def check_reporting_mode():
    return set_reporting_mode(EXE_MODE)


def set_reporting_mode(exe_mode):
    if exe_mode:
        # Silence warnings
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=UserWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=DeprecationWarning)

        return di.display_html(
            '<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>',
            raw=True)
