import argparse
import sys
from pathlib import Path

# Use the bundled DataManager, which supports full-band dataset wrappers.
_local_dassl = Path(__file__).resolve().parent / "Dassl.pytorch"
if _local_dassl.is_dir():
    sys.path.insert(0, str(_local_dassl))

import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.houstonu
import datasets.indian_pines
import datasets.ksc
import datasets.paviac
import datasets.paviau
import datasets.salinas




import trainers.zsrobust
import trainers.advmaple
import trainers.Advzsclip
import trainers.advindependentVL
import trainers.fap

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed is not None:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

   
    cfg.DATASET.HSI = CN()
    cfg.DATASET.HSI.HSI_FILE = ""
    cfg.DATASET.HSI.GT_FILE = ""
    cfg.DATASET.HSI.HSI_KEY = ""
    cfg.DATASET.HSI.GT_KEY = ""
    cfg.DATASET.HSI.CLASS_NAMES = []
    cfg.DATASET.HSI.PATCH_SIZE = 13
    cfg.DATASET.HSI.STRIDE = 1
    cfg.DATASET.HSI.SPLIT_FILE = "split_{dataset}.json"
    cfg.DATASET.HSI.IGNORED_LABELS = [0]
    cfg.DATASET.HSI.TRAIN_RATIO = 0.6
    cfg.DATASET.HSI.VAL_RATIO = 0.2
    cfg.DATASET.HSI.FLIP = True
    cfg.DATASET.HSI.RADIATION = True
    cfg.DATASET.HSI.MIXTURE = True

    # Config for zsrobust 
    cfg.TRAINER.ZSROBUST = CN()
    cfg.TRAINER.ZSROBUST.PREC = "fp16"
    cfg.TRAINER.ZSROBUST.VISUAL_PROMPT_METHOD = "padding"
    cfg.TRAINER.ZSROBUST.PROMPT_SIZE = 30
    cfg.TRAINER.ZSROBUST.ADD_PROMPT_SIZE = 5
    
    # Config for ADVMaPLe
    cfg.TRAINER.ADVMAPLE = CN()
    cfg.TRAINER.ADVMAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.ADVMAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.ADVMAPLE.PREC = "fp16"  # fp16, fp32, am
    cfg.TRAINER.ADVMAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TRAINER.ADVMAPLE.CLASSIFIER = False
    
    # Config for Adv independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.ADVIVLP = CN()
    cfg.TRAINER.ADVIVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.ADVIVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.ADVIVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.ADVIVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.ADVIVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.ADVIVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

        # Config for FAP
    cfg.TRAINER.FAP = CN()
    cfg.TRAINER.FAP.N_CTX = 2  # number of context vectors
    cfg.TRAINER.FAP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.FAP.PREC = "fp16"  # fp16, fp32, am
    cfg.TRAINER.FAP.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TRAINER.FAP.CLASSIFIER = False
    
    #add adversarial attack cfg
    cfg.ATTACK = CN()
    cfg.ATTACK.PGD = CN()
    cfg.ATTACK.PGD.EPS= 1
    cfg.ATTACK.PGD.ALPHA= 1
    cfg.ATTACK.PGD.TRAIN_ITER= 2
    cfg.ATTACK.PGD.TEST_ITER= 100
    cfg.ATTACK.PGD.ADV_TERM= "ce"
    cfg.ATTACK.PGD.LAMBDA_1= 1.0
    # HSI PGD converts EPS and ALPHA from 1/255 units inside the attack.
    cfg.ATTACK.HSI_PGD = CN()
    cfg.ATTACK.HSI_PGD.EPS = 1
    cfg.ATTACK.HSI_PGD.ALPHA = 1
    cfg.ATTACK.HSI_PGD.NORM = "l_inf"
    cfg.ATTACK.HSI_PGD.LOWER = 0.0
    cfg.ATTACK.HSI_PGD.UPPER = 1.0
    cfg.ATTACK.HSI_PGD.BATCH_SIZE = 4
    cfg.ATTACK.HSI_PGD.TEST_BATCH_SIZE = 128
    cfg.ATTACK.HSI_PGD.TRAIN_ITER = 2
    cfg.ATTACK.HSI_PGD.TEST_ITER = 50
    cfg.ATTACK.HSI_PGD.ADV_TERM = "cos"
    cfg.ATTACK.HSI_PGD.LAMBDA_1 = 1.0




    
def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
