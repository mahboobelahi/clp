# ============================================================
# Decoder / Placement
# ============================================================
DECODER_KIND = "two_phase"          # "baseline" | "two_phase"
BOX_ORDER_POLICY = None#"volume_then_maxface"            # e.g. "volume_then_maxface" (non-GA only)
SPLIT_RATIO = 0.70

SUPPORT_REQUIRED = True
SUPPORT_MIN_RATIO = 0.80
SOFT_ROTATION = False               # allow soft rotation in decoder
ROTATION_MODE_SETTING = "both" #c1,six

# ============================================================
# Genetic Algorithm
# ============================================================
GA_TEST = False                     # evaluate Gen0 only
GA_EVOLVE = True                    # full NSGA-II evolution

POP_SIZE = 100
GENERATIONS = 10

is_ULO = False#True                       # enable Z2 (ULO objective)

# ============================================================
# Dataset / Results
# ============================================================
WD_DIR = [
    "BR-Original-baseline",
    "BR-Original-two_phase",
    "BR-Modified-NSGA2_bi",
    "BR-Modified-NSGA2_tri",
]

RESULTS_DIR_NAME = WD_DIR[2]        # default: BR-Modified-NSGA2_bi

ENABLE_TEST_CLASS = True#False # True
ENABLE_TEST_CASE = False

# ============================================================
# Debug / Visualization
# ============================================================
PLOT_PARTIAL_LAYOUT = False
PLOT_POP_EVALUATION = False

debug = False                       # verbose debug prints

# ============================================================
# Speed/robustness caps (tune later)
# ============================================================
MAX_EPS_KEEP = 120      # cap EP list after each placement
MAX_EP_PROBES = 80      # cap EPs evaluated per decision
MAX_ROT_TRIES = 6       # cap rotation options tried per group