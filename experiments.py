import json
import time
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import shutil

import wandb
from cost_aware_bo import generate_hps, log_metrics, update_dataset_new_run

from segmentation import (  # Importing DroneDataset to prevent pickle error
    DroneDataset, main)

parser = ArgumentParser()
parser.add_argument("--exp-name", type=str, required=True, help="Specifies a unique experiment name")
parser.add_argument("--trial", type=int, help="The trial number", default=0)
parser.add_argument("--init-eta", type=float, help="Initial ETA", default=1)
parser.add_argument("--decay-factor", type=float, help="Decay factor", default=0.95)
parser.add_argument("--acqf", type=str, help="Acquisition function", choices=["EEIPU", "EIPS", "CArBO", "EI", "RAND"], default="EI")
parser.add_argument("--cache-root", type=Path, default=".cachestore", help="Cache directory")

args, _ = parser.parse_known_args()

dataset = {}
with open("segmentation/sampling-range.json") as f:
    hp_sampling_range = json.load(f)
    
params = {
    # "decay_factor": 0.95,
    # "init_eta": 0.05,
    "n_trials": 10,
    "n_iters": 40,
    "alpha": 9,
    "norm_eps": 1,
    "epsilon": 0.1,
    "batch_size": 1,
    "normalization_bounds": [0, 1],
    "cost_samples": 1000,
    "n_init_data": 10,
    "prefix_thresh": 10000000,
    "warmup_iters": 10,
    "use_pref_pool": 1,
    "verbose": 1,
    "rand_seed": 42,
    "total_budget": 17000,
    "budget_0": 7000
}

args_dict = deepcopy(vars(args))
params.update(args_dict)
    
wandb.init(
        entity="cost-bo",
        project="memoised-realworld-exp",
        group=f"{args.exp_name}|-acqf_{args.acqf}|-dec-fac_{args.decay_factor}"
                f"|init-eta_{args.init_eta}",
        name=f"{time.strftime('%Y-%m-%d-%H%M')}-trial-number_{args.trial}",
        config=params,
    )

consumed_budget, total_budget, init_budget = 0, params['total_budget'], params['budget_0']
i=0
warmup = True
while consumed_budget < total_budget:
    tic = time.time()
    
    if consumed_budget > init_budget and warmup:
        warmup = False
        params['n_init_data'] = i + 0

    new_hp_dict, logging_metadata = generate_hps(
        dataset,
        hp_sampling_range,
        iteration=i,
        params=params,
        consumed_budget=consumed_budget,
        acq_type=args.acqf, 
    )
    
    score_miou, score_miou_crf, cost_per_stage = main(new_hp_dict)
    
    consumed_budget += sum(cost_per_stage)

    dataset = update_dataset_new_run(dataset, new_hp_dict, cost_per_stage, score_miou_crf, logging_metadata["x_bounds"], args.acqf)
    
    print(f"\n\n[{time.strftime('%Y-%m-%d-%H%M')}]    Iteration-{i} [acq_type: {args.acqf}] Trial No. #{args.trial} Runtime: {time.time()-tic} Consumed Budget: {consumed_budget}")
    eta = (total_budget - consumed_budget) / (total_budget - params['budget_0'])
    log_metrics(dataset, logging_metadata, verbose=params["verbose"], iteration=i, trial=args.trial, acqf=args.acqf, eta=eta)
    i += 1

# Clean up cache
shutil.rmtree(args.cache_root)