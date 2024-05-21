import yaml

CONFIG_PATH = "./Configs/accent_adapter_config/accent_adapter_modules.yaml"
CONFIG = yaml.safe_load(open(CONFIG_PATH))

N_S = CONFIG["model"]["n_s"]
N_P = CONFIG["model"]["n_p"]
N_A = CONFIG["model"]["n_a"]
N_O = CONFIG["model"]["n_o"]
NUM_CLASSES = CONFIG["model"]["num_classes"]
ENC_HIDDEN_SIZES = CONFIG["model"]["enc_hidden_sizes"]
DEC_HIDDEN_SIZES = CONFIG["model"]["dec_hidden_sizes"]
A2O_HIDDEN_SIZES = CONFIG["model"]["a2o_hidden_sizes"]
O2A_HIDDEN_SIZES = CONFIG["model"]["o2a_hidden_sizes"]

MLP_CONFIG = CONFIG["model"]["layer_conf"]["MLP"]
ACCENT_PREDICTOR_CONFIG = CONFIG["model"]["layer_conf"]["accent_predictor"]
ENCODER_CONFIG = CONFIG["model"]["layer_conf"]["encoder"][0]
DECODER_CONFIG = CONFIG["model"]["layer_conf"]["decoder"][0]
A2O_DISENTANGLER_CONFIG = CONFIG["model"]["layer_conf"]["a2o_disentangler"][0]
O2A_DISENTANGLER_CONFIG = CONFIG["model"]["layer_conf"]["o2a_disentangler"][0]
