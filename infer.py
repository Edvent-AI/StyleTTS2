import numpy as np
import glob

from models import *
from utils import *
import argparse
from infer_utils import *


def get_args():
    args = argparse.ArgumentParser()

    args.add_argument("--outdir", type=str, required=False, default="./outdir")
    args.add_argument(
        "--config", type=str, required=False, default="./Configs/config_libritts.yml"
    )
    args.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="./Models/libri/libri_epochs_2nd_00020.pth",
    )
    args.add_argument(
        "--ref_audios",
        default="./Models/libri/reference_audio",
    )
    args.add_argument("--device", default="cpu")

    return args.parse_args()


if __name__ == "__main__":
    args = get_args()

    device = args.device

    model, sampler, model_params = load_modules(
        args.config, args.model_path, args.device
    )

    ref_audios = glob.glob(f"{args.ref_audios}/*.wav")

    for ref_aud in ref_audios:
        wav_name = os.path.basename(ref_aud)

        _, ref_s, ref_p = compute_style(ref_aud, args.device, model)

        outdir = os.path.join(args.outdir, wav_name[:-4])
        os.makedirs(outdir, exist_ok=True)

        np.save(outdir + "/ref_s.npy", ref_s)
        np.save(outdir + "/ref_p.npy", ref_p)
