import logging
import os
import random
import warnings
import platform
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np

import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Context, Parameters, FitIns, EvaluateIns, NDArrays, Scalar, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager

# Ensure Ultralytics does not use HUB (prevents import issues)
os.environ["ULTRALYTICS_HUB"] = "0"
from ultralytics import YOLO
from my_project.task import download_model, should_checkpoint, IS_WINDOWS, OS_NAME
from my_project.get_set_model import (NUM_CLASSES_MODEL_YAML, get_weights,
                                      learning_checksum, set_weights, warm_start_head)

from utils.logging_setup import configure_logging
from utils.metrics_logger import MetricsLogger, aggregate_client_metrics

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("server", "logs/server.log")

# Log OS detection
logger.info(f"[Server] Detected operating system: {OS_NAME}, IS_WINDOWS={IS_WINDOWS}")

# Constants
MODEL_PATH = "models/yolov8s.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
DEFAULT_BATCH_ID_RANGE = (1, 10)
DEFAULT_NUM_ROUNDS = 3

def round_config(server_round: int, num_rounds: int, local_epochs: int, *,
                 plots_every_round: bool = False, optimizer: str = "auto",
                 lr0: float = 0.0, mosaic: float = -1.0,
                 freeze_round1: int = 0, fix_bn_from_round: int = 0,
                 imgsz: int = 640) -> Dict[str, Scalar]:
    """What every client is told about THIS round.

    Shared across clients on purpose, and safe to share -- unlike ``batch_id``, which
    is per-vehicle and whose sharing was the B9 bug. Every value here is a property of
    the round, so "one FitIns for everyone" and "each client gets its own" agree.

    ``plots``: Ultralytics draws labels.jpg, train_batch*.jpg and, at final_eval, the
    confusion matrix and PR/F1 curves. Measured at roughly a fifth of a one-epoch
    round on this hardware, per client, every round -- and the client passes
    ``exist_ok=True``, so each round wrote them into the directory the next round
    overwrote. ``pipeline/train_artifacts.py`` serves that directory and its docstring
    already records that it holds only the last round. So the earlier rounds' pictures
    were drawn, paid for, and destroyed unread. Draw them on the round whose output
    actually survives.

    ``lr0`` is inert unless ``optimizer`` is set: ``optimizer="auto"`` replaces lr0
    with ``0.002*5/(4+nc)``. The client warns when it is handed that combination.

    ``freeze``: how many leading layers each client freezes, and it is sent as a
    per-round value because it is only wanted on round 1. ``.load()`` transfers 349 of
    355 tensors and cannot transfer the three classification convolutions, so the head
    starts part-warmed at best -- 9 of 13 classes from COCO, the other four random. Round
    1 therefore backpropagates that head's error into a backbone that already knew what a
    car looks like, and this project has measured the damage: the warm-started model
    scored 0.2582 on the holdout untrained and 0.2073 after two rounds. Freezing
    ``model.0.``..``model.9.`` for round 1 only lets the head settle against fixed
    features, which is what three independent papers arrive at -- linear-probe-then-
    fine-tune (arXiv:2306.03937), FedBABU (arXiv:2106.06042), and FedSTO's first stage on
    this very dataset (arXiv:2310.17097).

    What ``freeze=10`` actually does, MEASURED on one epoch of batch_1 rather than
    assumed -- and it is stronger than "no gradient":

    ```
    backbone model.0-9   weight   changed  0/54     head model.10-22  weight   changed 66/67
    backbone model.0-9   bias     changed  0/27     head model.10-22  bias     changed 36/36
    backbone model.0-9  running_var changed 0/27    head model.10-22 running_var changed 30/30
    requires_grad=False on 82/184 parameter tensors
    ```

    The statistics do not drift either, because ``_model_train``
    (``engine/trainer.py:701-707``) puts every frozen layer's ``BatchNorm2d`` into
    ``eval()`` -- "Freeze BN stat", in their words. So round 1 sends back a backbone that
    is *bit-identical* on every vehicle, which makes its aggregation an average of
    identical tensors and turns round 1 into a purely head-fitting round, federated only
    where the disagreement is. That is precisely the arrangement FedBABU and LP-FT
    describe, arrived at by the library's own implementation detail.

    ``freeze=N`` means ``model.0.``..``model.{N-1}.`` plus ``.dfl``
    (``engine/trainer.py:330-352``), so 10 is the backbone of YOLOv8s, and a value large
    enough to leave nothing trainable raises instead of training nothing.
    """
    return {
        "local_epochs": local_epochs,
        "server_round": server_round,
        "total_rounds": num_rounds,
        "plots": bool(plots_every_round) or server_round >= num_rounds,
        "optimizer": optimizer,
        "lr0": lr0,
        "mosaic": mosaic,
        # Round 1 only. A freeze that stayed on would federate a frozen backbone for
        # the whole run, which is a different experiment wearing this one's name.
        "freeze": int(freeze_round1) if server_round == 1 else 0,
        # FixBN: on from the named round to the end of the run, never for one round.
        # A warm-up is the method, not an implementation detail -- the statistics have
        # to be worth pinning before they are pinned, and 1 pins the COCO initial ones.
        "fix_bn": bool(fix_bn_from_round) and server_round >= int(fix_bn_from_round),
        # The federation's input resolution. Sent rather than hardcoded so the
        # holdout and the centralised baseline -- which already take it as a flag --
        # can be scored at the same size. A federation at 1024 measured against a
        # ceiling at 640 is not a comparison.
        "imgsz": int(imgsz),
    }


def evaluate_floor(fraction_evaluate: float, min_clients: int) -> int:
    """How few clients may evaluate a round -- NOT the same number as for fit.

    Flower picks `max(int(available * fraction_evaluate), min_evaluate_clients)`. This
    project passed `min_evaluate_clients = min_clients`, and the pipeline sets
    `min_clients = n_vehicles`, so the floor was the whole fleet and
    `max(int(6 * 0.34), 6) = 6`. **`fraction_evaluate` could not reduce anything.** It was
    read from run_config, logged, passed to the strategy, and inert -- the shape of silent
    no-op this project's CLAUDE.md keeps a table of.

    Measured 2026-09-26: a 2-round run at `fraction_evaluate=0.34` still performed 12
    client self-evaluations, which is 6 clients x 2 rounds, full participation.

    The fit minimum answers "how many vehicles must train for a round to be worth
    aggregating"; the evaluate minimum answers "how many must self-score for the row in
    metrics.csv to mean anything". Tying the second to the first was the bug. At
    `fraction_evaluate = 1.0` this returns `min_clients`, so nothing already measured moves.
    """
    return max(1, min(int(min_clients), round(float(fraction_evaluate) * int(min_clients))))


class BatchAssignmentMixin:
    """Everything this project needs from a strategy, independent of how it aggregates.

    Mixed in *before* a Flower strategy, so `type("X", (BatchAssignmentMixin, FedAdam), {})`
    gives FedAdam's aggregation with this project's shard assignment, checksum
    logging, checkpointing and metrics rows. Every override calls super(), which is
    what makes it compose rather than replace.

    Written as a mixin because these four behaviours were welded to FedAvg by
    inheritance, and copying the class to try another aggregator would have meant two
    copies of the shard-assignment logic -- which has already produced two silent
    failures in this repo (B9: one shared FitIns mutated for every client; B7:
    checkpointing skipped every round).

    Attributes:
        batch_id_range (tuple): Min and max range for batch IDs (inclusive)
        client_to_batch_id (dict): Maps each client to its shard, for the whole run
            (FL data locality: a client keeps the same shard every round)
    """

    def __init__(
        self,
        *,
        batch_id_range: tuple = DEFAULT_BATCH_ID_RANGE,
        proximal_mu: float = 0.0,
        num_rounds: int = DEFAULT_NUM_ROUNDS,
        checkpoint_dir: str = "checkpoints",
        save_every: int = 1,
        server_ema: float = 0.0,
        **kwargs: Any,
    ):
        # Everything else belongs to whichever strategy is mixed in underneath, and
        # they do not agree on a signature: FedAdam takes eta and tau, FedAvgM takes
        # server_momentum, FedAvg takes neither. Passing the rest through is what
        # lets one __init__ serve all of them.
        super().__init__(**kwargs)

        self.batch_id_range = batch_id_range
        self.client_to_batch_id: Dict[str, int] = {}
        self.client_os_info: Dict[str, str] = {}  # Track client OS information
        self.num_rounds = num_rounds
        self.metrics_logger = MetricsLogger()  # writes logs/metrics.csv

        # Global-model checkpointing: persist the aggregated model to disk so the
        # federated result survives process exit. Lazily instantiate one YOLO to
        # hold/save weights (created on first save to avoid a redundant load).
        self.checkpoint_dir = checkpoint_dir
        self.save_every = max(int(save_every), 1)
        self._save_model = None
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # proximal_mu > 0 turns on FedProx-style proximal regularization on clients.
        self.proximal_mu = float(proximal_mu)

        # Server-side EMA across rounds. 0.0 = off, which is the default, so this
        # changes no number until it is asked for. See _apply_server_ema.
        self.server_ema = float(server_ema)
        self._ema_weights: Optional[NDArrays] = None
        self._ema_rounds = 0
        logger.info(
            f"[Server] {type(self).__name__} initialized with batch_id_range={batch_id_range}, "
            f"aggregation={type(self).__mro__[2].__name__} (proximal_mu={self.proximal_mu})"
        )
    
    def _get_unused_batch_id(self, client_id: str) -> int:
        """
        Return this client's shard, assigning a free one on first sight.

        A client keeps the same shard for the whole run — that is the FL data
        locality premise, not an optimisation.

        Args:
            client_id: The client's identifier

        Returns:
            int: The batch ID this client owns
        """
        # Check if client already has a batch_id assigned
        if client_id in self.client_to_batch_id:
            logger.debug(f"[Server] Client {client_id} already has batch_id={self.client_to_batch_id[client_id]}")
            return self.client_to_batch_id[client_id]
            
        min_id, max_id = self.batch_id_range
        # Derive what is taken from what clients actually hold. A separate
        # used_batch_ids set was cleared every round while client_to_batch_id was
        # not, so a client joining in a later round could be handed a shard another
        # client was already training.
        taken = set(self.client_to_batch_id.values())
        available_ids = set(range(min_id, max_id + 1)) - taken

        if not available_ids:
            logger.warning(
                f"[Server] More clients than shards in range {min_id}-{max_id}; "
                f"{client_id} will share a shard with another client."
            )
            available_ids = set(range(min_id, max_id + 1))

        batch_id = random.choice(sorted(available_ids))
        self.client_to_batch_id[client_id] = batch_id
        logger.debug(f"[Server] Assigned new batch_id={batch_id} to client {client_id}")
        return batch_id
    
    def _save_global_model(self, weights, server_round: int) -> None:
        """
        Save the aggregated global weights as a self-contained YOLO checkpoint.

        Loads the ndarray weights into a held YOLO model (via set_weights, which
        now includes BatchNorm buffers) and writes both a per-round checkpoint and
        a stable ``global_last.pt`` pointer. Failures are logged, never fatal —
        a checkpointing error must not abort the federation.
        """
        try:
            if self._save_model is None:
                self._save_model = YOLO(NUM_CLASSES_MODEL_YAML).load(MODEL_PATH)
            if not set_weights(self._save_model.model, weights):
                logger.error(f"[Server] Round {server_round}: set_weights failed; skipping checkpoint.")
                return
            round_path = os.path.join(self.checkpoint_dir, f"global_round_{server_round}.pt")
            last_path = os.path.join(self.checkpoint_dir, "global_last.pt")
            self._save_model.save(round_path)
            self._save_model.save(last_path)
            logger.info(f"[Server] Saved global checkpoint: {round_path} (and {last_path})")
        except Exception as e:
            logger.error(f"[Server] Failed to save global checkpoint at round {server_round}: {e}", exc_info=True)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure training by assigning unique batch_id to each client.
        
        Args:
            server_round: Current round number
            parameters: Model parameters to distribute
            client_manager: Manages available clients
            
        Returns:
            List of tuples containing client proxies and their fit instructions
        """
        logger.info(f"[Server] configure_fit: Round={server_round}. Assigning batch IDs to clients...")

        # Log parameter information for debugging
        weights = parameters_to_ndarrays(parameters)
        weights_checksum = learning_checksum(weights)
        logger.info(f"[Server] Sending parameters with checksum: {weights_checksum}")

        # Delegate to FedAvg for the initial instructions
        instructions = super().configure_fit(server_round, parameters, client_manager)
        logger.info(f"[Server] Configured {len(instructions)} clients for training in round {server_round}")

        updated_instructions = []
        for (client_proxy, fit_ins) in instructions:
            # COPY, do not mutate. FedAvg.configure_fit builds ONE FitIns and hands
            # the same object to every client, so writing batch_id into fit_ins.config
            # in this loop overwrote it for all of them — last client won and the whole
            # federation trained a single shard while the log claimed otherwise.
            fit_config = dict(fit_ins.config)
            try:
                # Assign a unique batch_id for this client
                batch_id = self._get_unused_batch_id(client_proxy.cid)
                
                # Insert batch_id into the config. local_epochs is supplied by
                # on_fit_config_fn (driven by run_config), so we do not override it here.
                fit_config["batch_id"] = batch_id
                # Tell the client how strongly to pull local weights back toward the
                # global model (0.0 => plain FedAvg, no proximal term).
                fit_config["proximal_mu"] = self.proximal_mu

                logger.info(
                    f"[Server] Assigning batch_id={batch_id} "
                    f"to client {client_proxy.cid} in round={server_round}"
                )

                # Recreate the FitIns with the updated config
                new_fit_ins = FitIns(parameters=fit_ins.parameters, config=fit_config)
                updated_instructions.append((client_proxy, new_fit_ins))

            except Exception as e:
                logger.error(
                    f"[Server] Failed to assign batch_id for client {client_proxy.cid}: {e}",
                    exc_info=True
                )
                updated_instructions.append((client_proxy, fit_ins))  # fallback

        return updated_instructions
    
    def configure_evaluate(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure evaluation by assigning unique batch_id to each client.
        
        Args:
            server_round: Current round number
            parameters: Model parameters to distribute
            client_manager: Manages available clients
            
        Returns:
            List of tuples containing client proxies and their evaluation instructions
        """
        logger.info(f"[Server] configure_evaluate: Round={server_round}. Assigning batch IDs...")

        # Log parameter information for debugging
        weights = parameters_to_ndarrays(parameters)
        weights_checksum = learning_checksum(weights)
        logger.info(f"[Server] Sending evaluation parameters with checksum: {weights_checksum}")

        # Use default Flower behavior
        instructions = super().configure_evaluate(server_round, parameters, client_manager)
        logger.info(f"[Server] Configured {len(instructions)} clients for evaluation in round {server_round}")

        updated_instructions = []
        for (client_proxy, eval_ins) in instructions:
            eval_config = dict(eval_ins.config)  # shared EvaluateIns — copy, see configure_fit
            try:
                # Use the same batch_id assignment logic
                batch_id = self._get_unused_batch_id(client_proxy.cid)
                
                eval_config["batch_id"] = batch_id
                logger.info(f"[Server] Assigned batch_id={batch_id} to client {client_proxy.cid} for evaluation.")

                # Create new EvaluateIns with updated config
                new_eval_ins = EvaluateIns(parameters=eval_ins.parameters, config=eval_config)
                updated_instructions.append((client_proxy, new_eval_ins))

            except Exception as e:
                logger.error(f"[Server] Failed to assign batch_id for evaluation: {e}", exc_info=True)
                updated_instructions.append((client_proxy, eval_ins))  # Fallback

        return updated_instructions

    def _apply_server_ema(self, weights: NDArrays, server_round: int) -> NDArrays:
        """Blend this round's aggregate with an exponential average of the previous ones.

        The federation ends every round on an essentially unsmoothed model, and the
        centralised ceiling it is compared against does not. Ultralytics builds a fresh
        ``ModelEMA`` per ``train()`` call with ``updates = 0``
        (``engine/trainer.py:404``) and ramps its decay as
        ``0.9999*(1 - exp(-updates/2000))`` (``utils/torch_utils.py:734``), so 88 client
        steps reach d = 0.043 and 352 reach 0.161, while the ceiling's 3 168 steps reach
        0.795. Averaging on the client cannot fix that -- the counter restarts every
        round by construction -- so the averaging has to happen where the rounds are,
        which is here.

        Bias-corrected, and that is not decoration. Without it, round 1 would return
        ``d * initial + (1 - d) * aggregate`` and drag the fleet back toward the model it
        started from; dividing by ``1 - d**t`` makes round 1 exactly the aggregate and
        lets the average build up from there. Same correction Adam uses, for the same
        reason.

        This replaces the broadcast *and* the checkpoint, because those are the same
        tensors and the method is only meaningful if the clients continue from the
        smoothed model -- that is what FedSWA (arXiv:2507.20016) and FedEMA
        (arXiv:2505.00318) do. It is therefore a different federation, not a reporting
        change, and it is off by default.

        Fails loudly on a shape or length change rather than silently restarting the
        average: a strategy that changed the tensor list mid-run would otherwise produce
        an EMA of two different architectures.
        """
        if self._ema_weights is None:
            # Zero, not the first aggregate. `m_t = d*m_{t-1} + (1-d)*w_t` divided by
            # `1 - d**t` is an exact average only if the accumulator starts empty;
            # seeding it with `w_1` instead makes the correction over-divide from round
            # 2 on, and a constant aggregate then comes back inflated -- which is how
            # the first draft of this method failed its own "nothing changed" test.
            self._ema_weights = [np.zeros_like(w, dtype=np.float64) for w in weights]
            self._ema_rounds = 0
            logger.info(f"[Server] Server-side EMA on at decay {self.server_ema}; "
                        f"round {server_round} is the aggregate itself.")

        if len(self._ema_weights) != len(weights) or any(
                e.shape != w.shape for e, w in zip(self._ema_weights, weights)):
            raise RuntimeError(
                f"[Server] server EMA state does not match this round's aggregate "
                f"({len(self._ema_weights)} vs {len(weights)} tensors). Refusing to "
                f"average two different models.")

        d = self.server_ema
        self._ema_rounds += 1
        self._ema_weights = [d * e + (1.0 - d) * w.astype(np.float64)
                             for e, w in zip(self._ema_weights, weights)]
        correction = 1.0 - d ** self._ema_rounds
        smoothed = [(e / correction).astype(w.dtype)
                    for e, w in zip(self._ema_weights, weights)]
        logger.info(f"[Server] Server-side EMA: round {server_round} blended at decay "
                    f"{d} over {self._ema_rounds} rounds (bias correction "
                    f"{correction:.4f}).")
        return smoothed

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitIns]],
        failures: List[Union[Tuple[ClientProxy, FitIns], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model parameters and training metrics from client training results.
        
        Args:
            server_round: Current round number
            results: Training results from each client
            failures: List of client failures
            
        Returns:
            Tuple of (aggregated parameters, metrics dict) 
        """
        if not results:
            logger.warning(f"[Server] No results to aggregate in round {server_round}")
            return None, {}
        
        # Track OS information from clients
        for client_proxy, fit_res in results:
            if "os" in fit_res.metrics:
                client_os = fit_res.metrics["os"]
                self.client_os_info[client_proxy.cid] = client_os
                logger.info(f"[Server] Client {client_proxy.cid} is running on {client_os}")
        
        # Log results and failures
        logger.info(f"[Server] Aggregating {len(results)} fit results and {len(failures)} failures")

        # Persist a weighted-average of client training metrics for this round.
        fit_metrics = aggregate_client_metrics(results)
        self.metrics_logger.log_round(server_round, "fit", fit_metrics, num_clients=len(results))

        # Call the parent's aggregation method
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if parameters is not None:
            # Calculate parameters checksum for verification
            weights = parameters_to_ndarrays(parameters)
            if self.server_ema > 0:
                weights = self._apply_server_ema(weights, server_round)
                parameters = fl.common.ndarrays_to_parameters(weights)
            weights_checksum = learning_checksum(weights)
            logger.info(f"[Server] Aggregated parameters with checksum: {weights_checksum}")

            # Persist the aggregated global model on the configured cadence and on
            # the final round, so the federated result is recoverable after exit.
            if should_checkpoint(server_round, self.save_every, self.num_rounds):
                self._save_global_model(weights, server_round)

            # Add checksum and OS counts to metrics for tracking
            metrics["weights_checksum"] = float(weights_checksum)
            metrics["server_os"] = OS_NAME
            
            # Count clients by OS
            os_counts = {}
            for os_name in self.client_os_info.values():
                os_counts[f"client_count_{os_name}"] = os_counts.get(f"client_count_{os_name}", 0) + 1
            
            # Add OS counts to metrics
            metrics.update({k: float(v) for k, v in os_counts.items()})
            
            # Log metrics
            logger.info(f"[Server] Round {server_round} metrics: {metrics}")
        else:
            logger.warning(f"[Server] No aggregated parameters produced in round {server_round}")
            
        return parameters, metrics
        
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitIns]],
        failures: List[Union[Tuple[ClientProxy, FitIns], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: Evaluation results from each client
            failures: List of client failures
            
        Returns:
            Tuple of (loss, metrics dict)
        """
        if not results:
            return None, {}
            
        # Track OS information from clients
        for client_proxy, eval_res in results:
            if "os" in eval_res.metrics:
                client_os = eval_res.metrics["os"]
                self.client_os_info[client_proxy.cid] = client_os
                logger.info(f"[Server] Client {client_proxy.cid} evaluated on {client_os}")
        
        # Persist a weighted-average of client evaluation metrics for this round.
        eval_metrics = aggregate_client_metrics(results)

        # Call the parent's aggregation method
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        self.metrics_logger.log_round(
            server_round, "evaluate", eval_metrics, num_clients=len(results), loss=loss
        )
        # Emit the run summary once the final round has been evaluated.
        if server_round >= self.num_rounds:
            self.metrics_logger.summary()

        # Add OS information to metrics
        metrics["server_os"] = OS_NAME
        
        # Count clients by OS
        os_counts = {}
        for os_name in self.client_os_info.values():
            os_counts[f"client_count_{os_name}"] = os_counts.get(f"client_count_{os_name}", 0) + 1
        
        # Add OS counts to metrics
        metrics.update({k: float(v) for k, v in os_counts.items()})
        
        return loss, metrics


class CustomBatchStrategy(BatchAssignmentMixin, FedAvg):
    """FedAvg with this project's shard assignment. The name three docs, the README
    and tests/test_batch_assignment.py refer to, kept so they keep meaning."""


# --------------------------------------------------------------------------
# Strategy registry
# --------------------------------------------------------------------------
#: name -> the Flower strategy it aggregates with. Probed rather than imported
#: directly, because which strategies exist varies by Flower version and an
#: ImportError at server start is a worse failure than an absent option.
_CANDIDATES = {
    "fedavg": "FedAvg",
    "fedprox": "FedAvg",          # FedAvg plus proximal_mu shipped to the clients
    "fedadam": "FedAdam",
    "fedyogi": "FedYogi",
    "fedadagrad": "FedAdagrad",
    "fedavgm": "FedAvgM",
    "fedmedian": "FedMedian",
    "fedtrimmedavg": "FedTrimmedAvg",
    "krum": "Krum",
    "bulyan": "Bulyan",
    "qfedavg": "QFedAvg",
    "faulttolerantfedavg": "FaultTolerantFedAvg",
}


def _available_strategies() -> Dict[str, type]:
    import flwr.server.strategy as flwr_strategies

    found = {}
    for name, attr in _CANDIDATES.items():
        base = getattr(flwr_strategies, attr, None)
        if base is not None:
            found[name] = base
    return found


STRATEGIES: Dict[str, type] = _available_strategies()


def _accepted_kwargs(cls: type) -> set:
    """Which keyword arguments a strategy's __init__ will actually take."""
    import inspect

    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return set()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return set(params) | {"__var_keyword__"}
    return {n for n in params if n != "self"}


def build_strategy(name: str, *, project_kwargs: Dict[str, Any],
                   common_kwargs: Dict[str, Any],
                   tuning_kwargs: Optional[Dict[str, Any]] = None):
    """Compose BatchAssignmentMixin with the named Flower strategy and instantiate it.

    An unknown name raises. Falling back to FedAvg would produce a run labelled
    FedAdam that is not one, and this repo's history is a catalogue of exactly that
    kind of quiet substitution.
    """
    key = (name or "fedavg").lower()
    if key not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {name!r}. Available in this Flower build: "
            f"{', '.join(sorted(STRATEGIES))}"
        )
    base = STRATEGIES[key]
    composed = type(f"{base.__name__}WithBatchAssignment", (BatchAssignmentMixin, base), {})

    accepted = _accepted_kwargs(base)
    takes_anything = "__var_keyword__" in accepted
    passed = {k: v for k, v in {**common_kwargs, **(tuning_kwargs or {})}.items()
              if takes_anything or k in accepted}
    dropped = sorted(set({**common_kwargs, **(tuning_kwargs or {})}) - set(passed))
    if dropped:
        # Said out loud: a silently dropped eta would make an FedAdam sweep report
        # identical numbers for every value and look like the knob does nothing.
        logger.info(f"[Server] {base.__name__} does not accept {dropped}; not passed")
    return composed(**project_kwargs, **passed)


def server_fn(context: Context):
    """
    Initialize the Flower server with a YOLO model and custom federated learning strategy.

    Hyperparameters are read from ``context.run_config`` (defined in pyproject.toml under
    ``[tool.flwr.app.config]``) instead of being hardcoded, so they can be overridden with
    ``flwr run --run-config "num_server_rounds=5 local_epochs=2"``.

    Returns:
        ServerAppComponents: Configured server components for federated learning
    """
    logger.info("[Server] Initializing YOLO model for FL...")

    # Read run configuration (with safe fallbacks).
    run_config = getattr(context, "run_config", {}) or {}
    num_rounds = int(run_config.get("num_server_rounds", DEFAULT_NUM_ROUNDS))
    fraction_fit = float(run_config.get("fraction_fit", 1.0))
    # Never set before, so FedAvg's 1.0 applied and every client re-scored itself on
    # its own split every round -- 13.8 % of wall clock (phase 0) spent on the metric
    # this project calls the flattering one, while the holdout is what gets reported.
    # Default stays 1.0 so this commit changes no numbers; the lever is now reachable.
    fraction_evaluate = float(run_config.get("fraction_evaluate", 1.0))
    local_epochs = int(run_config.get("local_epochs", 1))
    min_clients = int(run_config.get("min_clients", 2))
    # Strategy selection from run_config: any name in STRATEGIES.
    strategy_name = str(run_config.get("strategy", "fedavg")).lower()
    proximal_mu = float(run_config.get("proximal_mu", 0.0))
    if strategy_name == "fedprox" and proximal_mu <= 0:
        proximal_mu = 0.1  # sensible default when FedProx is requested without a mu
    if strategy_name != "fedprox":
        proximal_mu = 0.0  # the proximal term is FedProx's, not everyone's
    # Server-side optimiser knobs. Each is passed only to a strategy whose __init__
    # accepts it, so setting eta with FedAvg selected is reported, not silently lost.
    tuning = {k: float(run_config[k]) for k in
              ("eta", "eta_l", "beta_1", "beta_2", "tau", "server_momentum", "q_param")
              if k in run_config}
    checkpoint_dir = str(run_config.get("checkpoint_dir", "checkpoints"))
    save_every = int(run_config.get("save_every", 1))
    # How much of a round is spent on things that are not training. See fit_config_fn.
    plots_every_round = bool(run_config.get("plots_every_round", False))
    # Ultralytics' `optimizer="auto"` DISCARDS lr0 and substitutes its own, so lr0 is
    # inert until this is set to a real optimiser name. Default stays "auto" so this
    # commit changes no numbers; the client warns loudly if lr0 is set without it.
    optimizer_name = str(run_config.get("optimizer", "auto"))
    lr0 = float(run_config.get("lr0", 0.0))          # 0.0 = leave Ultralytics alone
    # Negative = unset. Flower's Scalar type has no None, and sending 0.0 would mean
    # "mosaic off" rather than "not specified" -- a default silently changed to its
    # opposite is exactly the failure this project keeps shipping.
    mosaic = float(run_config.get("mosaic", -1.0))
    # Round 1 only, and 0 = off, so this commit changes no numbers until it is set. 10 is
    # the YOLOv8s backbone; see round_config for why round 1 is the round that wants it.
    freeze_round1 = int(run_config.get("freeze_round1", 0))
    # Server-side EMA across rounds, 0.0 = off. A real federation change, not a
    # reporting one: the clients continue from the smoothed model. See
    # BatchAssignmentMixin._apply_server_ema.
    server_ema = float(run_config.get("server_ema", 0.0))
    # FixBN (arXiv:2303.06530): freeze the normalisation statistics from this round
    # on. 0 = off. Unlike FedBN it leaves ONE global model, so the holdout still
    # measures the model that was trained.
    fix_bn_from_round = int(run_config.get("fix_bn_from_round", 0))
    # BDD100K frames are 1280x720 and this trains at 640 by default, where a traffic
    # light is sub-20px. Published YOLOv8 numbers on this dataset: 0.470 mAP50 for
    # yolov8n at 640 against 0.625 for yolov8s at 1024. The pipeline scores the
    # holdout and the baseline at whatever this is, so the comparison stays fair.
    imgsz = int(run_config.get("imgsz", 640))
    # FedBN is a client-side filter -- the server still receives and averages every
    # tensor, clients simply decline the BatchNorm ones. Recorded here so the run log
    # says which federation this was, and so the caveat below is on the record.
    local_bn = bool(run_config.get("local_bn", False))
    logger.info(
        f"[Server] run_config -> num_rounds={num_rounds}, fraction_fit={fraction_fit}, "
        f"fraction_evaluate={fraction_evaluate}, local_epochs={local_epochs}, "
        f"min_clients={min_clients}, strategy={strategy_name}, "
        f"proximal_mu={proximal_mu}, local_bn={local_bn}, "
        f"freeze_round1={freeze_round1}, server_ema={server_ema}, "
        f"fix_bn_from_round={fix_bn_from_round}, imgsz={imgsz}"
    )
    if local_bn:
        logger.warning(
            "[Server] FedBN is ON. Each vehicle keeps its own BatchNorm, so there is "
            "no single global model: the checkpoint saved each round carries the "
            "AVERAGED BatchNorm, which is no vehicle's. A holdout score on it measures "
            "a model that never existed. Compare per-vehicle numbers, or re-estimate "
            "BatchNorm on the holdout, and say which."
        )

    # Check if model exists, otherwise download
    if not os.path.exists(MODEL_PATH):
        logger.info(f"[Server] Model not found at {MODEL_PATH}, downloading...")
        download_model()
        logger.info(f"[Server] Model successfully downloaded to {MODEL_PATH}")

    # Load YOLO's initial model
    try:
        # Same 13-class arch the clients build (client_app.py). Building from the
        # .pt alone gives an 80-class COCO head, which stops matching the moment a
        # client's train() rebuilds its head from data.yaml (nc=13). Pass the inner
        # DetectionModel, not the YOLO wrapper, so both sides key the state_dict
        # identically.
        model = YOLO(NUM_CLASSES_MODEL_YAML).load(MODEL_PATH)
        # .load() transfers 349 of 355 tensors; the six it cannot are the three
        # classification convolutions, whose shapes differ between 80 and 13 classes.
        # Those stay random unless warmed, and this model IS what round 1 broadcasts,
        # so a random head here is a random head on every client no matter what they
        # build locally. BDD100K shares most of its road classes with COCO.
        warmed = warm_start_head(model.model, YOLO(MODEL_PATH))
        logger.info(f"[Server] Head warm-started from COCO for {len(warmed)} classes: "
                    f"{', '.join(warmed) if warmed else 'none — head is random'}")
        initial_weights = get_weights(model.model)

        # Calculate initial checksum for tracking
        initial_checksum = sum(w.sum() for w in initial_weights if w.size > 0)
        logger.info(f"[Server] Initial model loaded with weights checksum: {initial_checksum}")
    except Exception as e:
        logger.error("[Server] Could not load YOLO model or extract weights!", exc_info=True)
        raise RuntimeError("Server cannot start without a valid YOLO model.") from e

    # Push local_epochs to every client each round via config callbacks.
    #
    # Safe to share one dict across clients, unlike batch_id (the B9 bug): every value
    # here is a property of the ROUND, not of the vehicle, so "the last value written
    # wins" and "every client gets the same value" are the same outcome.
    def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        return round_config(server_round, num_rounds, local_epochs,
                            plots_every_round=plots_every_round,
                            optimizer=optimizer_name, lr0=lr0, mosaic=mosaic,
                            freeze_round1=freeze_round1,
                            fix_bn_from_round=fix_bn_from_round, imgsz=imgsz)

    # Build the strategy through the registry: the mixin carries this project's
    # behaviour, the named Flower strategy carries the aggregation.
    strategy = build_strategy(
        strategy_name,
        project_kwargs=dict(
            batch_id_range=DEFAULT_BATCH_ID_RANGE,
            proximal_mu=proximal_mu,
            num_rounds=num_rounds,
            checkpoint_dir=checkpoint_dir,
            save_every=save_every,
            server_ema=server_ema,
        ),
        common_kwargs=dict(
            fraction_fit=fraction_fit,            # From run_config
            fraction_evaluate=fraction_evaluate,  # From run_config
            min_fit_clients=min_clients,          # From run_config
            min_evaluate_clients=evaluate_floor(fraction_evaluate, min_clients),
            min_available_clients=min_clients,    # Minimum clients needed to start FL
            on_fit_config_fn=fit_config_fn,
            on_evaluate_config_fn=fit_config_fn,
            initial_parameters=fl.common.ndarrays_to_parameters(initial_weights),
        ),
        tuning_kwargs=tuning,
    )

    # Configure server
    server_config = ServerConfig(num_rounds=num_rounds)

    logger.info(f"[Server] strategy={strategy_name} ({type(strategy).__name__}) "
                f"configured for {num_rounds} rounds; "
                f"available: {', '.join(sorted(STRATEGIES))}")
    logger.info(f"[Server] Server running on {OS_NAME} is ready for clients")
    
    return ServerAppComponents(
        strategy=strategy, 
        config=server_config
    )


app = ServerApp(server_fn=server_fn)