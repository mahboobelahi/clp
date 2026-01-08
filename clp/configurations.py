# ============================================================
# Decoder / Placement
# ============================================================
POLICIES = [

    "volume_then_maxface",
    "customer_then_volume_maxface",
    "maxface_then_volume",
    "min_height_then_volume",
    "random_tiebreak"
]

DECODERS = ["baseline", "two_phase"]

MODES = ["c1","six","both"]

DECODER_KIND = DECODERS[1]          # "baseline" | "two_phase"

DEBUG_F = [True,False]
# e.g. "volume_then_maxface" (non-GA only)
BOX_ORDER_POLICY = POLICIES[1]
SPLIT_RATIO = 0.80

SUPPORT_REQUIRED = True
SUPPORT_MIN_RATIO = 0.80
SOFT_ROTATION = False               # allow soft rotation in decoder
ROTATION_MODE_SETTING = MODES[0]

# ============================================================
# Genetic Algorithm
# ============================================================
GA_TEST = False                     # evaluate Gen0 only
GA_EVOLVE = True                    # full NSGA-II evolution

POP_SIZE = 100
GENERATIONS = 10

is_ULO = True                       # enable Z2 (ULO objective)

# ============================================================
# Dataset / Results
# ============================================================
WD_DIR = [
    "BR-Original-baseline",
    "BR-Original-two_phase",
    "BR-Modified-NSGA2_bi",
    "BR-Modified-NSGA2_tri",
    "BR-Modified-NSGA2_tri_cust",
]

RESULTS_DIR_NAME = WD_DIR[4]        # default: BR-Modified-NSGA2_bi

ENABLE_TEST_CLASS = DEBUG_F[1]  
ENABLE_TEST_CASE = DEBUG_F[1] 

# ============================================================
# Debug / Visualization
# ============================================================
PLOT_PARTIAL_LAYOUT = DEBUG_F[1] 
PLOT_POP_EVALUATION = DEBUG_F[1] 

debug = DEBUG_F[1]                        # verbose debug prints

# ============================================================
# Speed/robustness caps (tune later)
# ============================================================
MAX_EPS_KEEP = 120      # cap EP list after each placement
MAX_EP_PROBES = 80      # cap EPs evaluated per decision
MAX_ROT_TRIES = 6       # cap rotation options tried per group
