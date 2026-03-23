#!/usr/bin/env python3
import os
from datetime import datetime
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(BASE_PATH).resolve().parents[1]

OUTPUTS_DIR = os.environ.get(
    "SHINKA_WFA_EXPERIMENT_OUTPUTS_DIR",
    str(REPO_ROOT / "experiments" / "shinka-evolve" / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")),
)

MODEL_NAMES = os.environ.get("SHINKA_WFA_LLM_MODELS", "local-openai/gpt-oss-120b")
MODEL_NAMES = MODEL_NAMES.split(",")
assert len(MODEL_NAMES) > 0, "SHINKA_WFA_LLM_MODELS must be set"

META_MODEL_NAMES = os.environ.get("SHINKA_WFA_META_LLM_MODELS", None)
META_MODEL_NAMES = META_MODEL_NAMES.split(",") if META_MODEL_NAMES else None
NOVELTY_MODEL_NAMES = os.environ.get("SHINKA_WFA_NOVELTY_LLM_MODELS", None)
NOVELTY_MODEL_NAMES = NOVELTY_MODEL_NAMES.split(",") if NOVELTY_MODEL_NAMES else None
EMBEDDING_MODEL_NAME = os.environ.get("SHINKA_WFA_EMBEDDING_MODEL", None)

TASK_NAME = os.environ.get("SHINKA_WFA_TASK_NAME", "simple_task")
TASK_DIR = os.environ.get("SHINKA_WFA_TASK_DIR", f"{BASE_PATH}/{TASK_NAME}")

os.environ["SHINKA_PROMPT_ROOT"] = f"{BASE_PATH}/prompts"
os.environ["SHINKA_EXPERIMENT_ROOT"] = TASK_DIR

EVALUATE_PATH = os.environ.get(
    "SHINKA_WFA_EVALUATE_PATH",
    str(REPO_ROOT / "tools" / "evaluator" / "evaluate.py"),
)

#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

job_config = LocalJobConfig(eval_program_path=EVALUATE_PATH)

strategy = os.environ.get("SHINKA_WFA_STRATEGY", "weighted")

if strategy == "uniform":
    # 1. Uniform from correct programs
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=0.0,
        exploitation_ratio=1.0,
    )
elif strategy == "hill_climbing":
    # 2. Hill Climbing (Always from the Best)
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=100.0,
        exploitation_ratio=1.0,
    )
elif strategy == "weighted":
    # 3. Weighted Prioritization
    parent_config = dict(
        parent_selection_strategy="weighted",
        parent_selection_lambda=10.0,
    )
elif strategy == "power_law":
    # 4. Power-Law Prioritization
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=1.0,
        exploitation_ratio=0.2,
    )
elif strategy == "power_law_high":
    # 4. Power-Law Prioritization
    parent_config = dict(
        parent_selection_strategy="power_law",
        exploitation_alpha=2.0,
        exploitation_ratio=0.2,
    )
elif strategy == "beam_search":
    # 5. Beam Search
    parent_config = dict(
        parent_selection_strategy="beam_search",
        num_beams=10,
    )

NUM_INSPIRATIONS = int(os.environ.get("SHINKA_WFA_NUM_INSPIRATIONS", 4))
NUM_TOP_K_INSPIRATIONS = int(os.environ.get("SHINKA_WFA_NUM_TOP_K_INSPIRATIONS", 2))
MIGRATION_INTERVAL = int(os.environ.get("SHINKA_WFA_MIGRATION_INTERVAL", 10))
MIGRATION_RATE = float(os.environ.get("SHINKA_WFA_MIGRATION_RATE", 0.1))
NUM_ISLANDS = int(os.environ.get("SHINKA_WFA_NUM_ISLANDS", 4))
ARCHIVE_SIZE = int(os.environ.get("SHINKA_WFA_ARCHIVE_SIZE", 40))
ELITE_SELECTION_RATIO = float(os.environ.get("SHINKA_WFA_ELITE_SELECTION_RATIO", 0.3))
ISLAND_ELITISM = os.environ.get("SHINKA_WFA_ISLAND_ELITISM", "true").lower() == "true"

db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=NUM_ISLANDS,
    archive_size=ARCHIVE_SIZE,
    # Inspiration parameters
    elite_selection_ratio=ELITE_SELECTION_RATIO,
    num_archive_inspirations=NUM_INSPIRATIONS,
    num_top_k_inspirations=NUM_TOP_K_INSPIRATIONS,
    # Island migration parameters
    migration_interval=MIGRATION_INTERVAL,
    migration_rate=MIGRATION_RATE,  # chance to migrate program to random island
    island_elitism=ISLAND_ELITISM,  # Island elite is protected from migration
    **parent_config,
)

SYSTEM_PATH = os.environ.get("SHINKA_EXPERIMENT_SYSTEM_PATH", f"{TASK_DIR}/SYSTEM.md")
search_task_sys_msg = Path(SYSTEM_PATH).read_text()

NUM_GENERATIONS = int(os.environ.get("SHINKA_WFA_NUM_GENERATIONS", 100))
PATCH_TYPE_PROBS = os.environ.get("SHINKA_WFA_PATCH_TYPE_PROBS", "0.6,0.3,0.1")
PATCH_TYPE_PROBS = [float(p) for p in PATCH_TYPE_PROBS.split(",")]
MAX_PARALLEL_JOBS = int(os.environ.get("SHINKA_WFA_MAX_PARALLEL_JOBS", 3))

MAX_TOKENS = int(os.environ.get("SHINKA_WFA_MAX_TOKENS", 80000))
MAX_LLM_TOKENS = int(os.environ.get("SHINKA_WFA_MAX_LLM_TOKENS", MAX_TOKENS))
MAX_META_TOKENS = int(os.environ.get("SHINKA_WFA_MAX_META_TOKENS", MAX_TOKENS))
MAX_NOVELTY_TOKENS = int(os.environ.get("SHINKA_WFA_MAX_NOVELTY_TOKENS", MAX_TOKENS))

INITIAL_PATH = os.environ.get("SHINKA_WFA_INITIAL_PATH", f"{TASK_DIR}/initial.py")

evo_config = EvolutionConfig(
    task_sys_msg=search_task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=PATCH_TYPE_PROBS,
    num_generations=NUM_GENERATIONS,
    max_parallel_jobs=MAX_PARALLEL_JOBS,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=MODEL_NAMES,
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],
        reasoning_efforts=["auto", "low", "medium", "high"],
        max_tokens=MAX_LLM_TOKENS,
    ),
    meta_rec_interval=10,
    meta_llm_models=META_MODEL_NAMES,
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=MAX_META_TOKENS),
    embedding_model=EMBEDDING_MODEL_NAME,
    code_embed_sim_threshold=0.995,
    novelty_llm_models=NOVELTY_MODEL_NAMES,
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=MAX_NOVELTY_TOKENS),
    llm_dynamic_selection="ucb1",
    llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
    init_program_path=INITIAL_PATH,
    results_dir=OUTPUTS_DIR,
)


try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def main():
    if wandb is not None:
        wandb.init()

    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()

if __name__ == "__main__":
    results_data = main()
