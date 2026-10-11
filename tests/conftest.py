"""Shared test setup: import the app modules from the project root, quietly."""
import logging
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# chem_v4 imports Streamlit; outside `streamlit run` it logs "no runtime"
# warnings on every cached call — noise, not failures.
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
for name in ("streamlit.runtime.caching.cache_data_api",
             "streamlit.runtime.scriptrunner_utils.script_run_context",
             "streamlit.runtime.state.session_state_proxy"):
    logging.getLogger(name).setLevel(logging.ERROR)
