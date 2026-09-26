import logging
import math
import os
import warnings
import torch
import platform
from flwr.common import Code, Status
from flwr.client import ClientApp, Client
from flwr.common import Context, FitIns, FitRes, EvaluateIns, EvaluateRes, Parameters, RecordSet

from ultralytics import YOLO
from my_project.task import (
    download_model,
    get_data_yaml_path,
    materialize_data_yaml,
    count_shard_examples,
    get_optimal_batch_size,
    IS_WINDOWS,
    OS_NAME
)  # Import OS detection
import urllib
from my_project.get_set_model import (NUM_CLASSES_MODEL_YAML, batchnorm_keys, get_weights,
                                      set_weights, warm_start_head)
from my_project.trainers import fixbn_trainer
from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("client", "logs/client.log")

# Log OS detection information
logger.info(f"[Client] Detected operating system: {OS_NAME}, IS_WINDOWS={IS_WINDOWS}")

# Constants
DEFAULT_BATCH_ID_RANGE = (1, 10)
# Ultralytics' nominal batch size: it accumulates gradients up to this many
# images before stepping the optimizer (ultralytics default `nbs`).
NOMINAL_BATCH_SIZE = 64
DEFAULT_IMAGE_SIZE = 640

class FlowerClient(Client):

    def __init__(self,
                model_path: str,
                client_state : RecordSet,
                local_epochs : int,
                batch_id_range: tuple = DEFAULT_BATCH_ID_RANGE,  # Using constant
                cache: str = "",
                local_bn: bool = False,
                ):

        super().__init__()
        self.model_path = model_path
        self.client_state = client_state
        self.local_epochs = local_epochs
        self.batch_id_range = batch_id_range
        self.batch_id = None
        # FedBN. Per-RUN, not per-round: it changes what the federation is, so it
        # belongs in run_config rather than in a FitIns the server could vary.
        self.local_bn = bool(local_bn)
        # "" | "ram" | "disk". Ultralytics wants False, not "", for "no cache".
        self.cache = cache or False
        
        # Decide GPU/CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"[Client] Initializing FlowerClient with model={model_path} on {self.device}")
        # Load YOLO model with valid initial weights
        try:
            if not os.path.exists(self.model_path):
                logger.warning("[Client] Model weights not found. Downloading default YOLOv8 weights.")
                download_model()  # Ensure we have some valid weights to avoid shape mismatch
            
            # Build the 13-class architecture from the yaml, then load the COCO
            # weights into it. Constructing from the .pt alone gives an 80-class
            # head; `model.nc = 13` only renames an attribute, it does not rebuild
            # the head, so the server and client end up with different archs the
            # moment Ultralytics rebuilds the head from data.yaml during train().
            # The server builds the same arch (server_app.py) — set_weights'
            # strict=True is what enforces that they agree.
            self.yolo = YOLO(NUM_CLASSES_MODEL_YAML).load(self.model_path)
            # The head the server broadcasts overwrites this one every round, so this
            # matters only for the shape of the model before the first set_weights.
            # Done anyway, and through the same helper, so the two sides cannot drift:
            # a client whose head is warmed differently from the server's is exactly
            # the kind of asymmetry set_weights' strict=True exists to catch.
            warm_start_head(self.yolo.model, YOLO(self.model_path))

            # Computed once: the architecture does not change between rounds, and
            # walking every module each round would be work for an unchanging answer.
            self._bn_keys = batchnorm_keys(self.model) if self.local_bn else set()
            if self.local_bn:
                logger.info(f"[Client] FedBN on: {len(self._bn_keys)} BatchNorm tensors "
                            f"stay local, the rest come from the aggregate.")
                if not self._bn_keys:
                    raise ValueError(
                        "local_bn was requested but no BatchNorm layers were found. "
                        "That would run plain FedAvg while the log says FedBN.")

            logger.info("[Client] YOLO model loaded successfully.")
        except Exception as e:
            logger.error("[Client] Failed to load YOLO model!", exc_info=True)
            self.yolo = None
            self._bn_keys = set()

    @property
    def model(self):
        """The live DetectionModel.

        Ultralytics rebinds ``yolo.model`` to the trained checkpoint at the end of
        ``train()``. Caching it in ``__init__`` left an orphan: the client returned
        the weights it was *sent*, never the ones it *trained*, so FedAvg averaged
        identical inputs and the federation learned nothing. Reading through means
        every call site sees the current module, including ones added later.
        """
        return self.yolo.model if self.yolo is not None else None

    @property
    def trained_model(self):
        """The module the optimizer actually stepped -- what FedAvg must average.

        ``model`` above is the right thing to *receive* into and the wrong thing to
        *send*. Ultralytics' rebind at the end of ``train()`` does not hand back the
        trained module; it reloads a checkpoint (``engine/model.py:828``), and the
        checkpoint holds only ``deepcopy(ema).half()`` -- ``"model"`` is written as
        ``None`` (``engine/trainer.py:725,740``) and the loader prefers ``ckpt["ema"]``
        (``nn/tasks.py:1899``). Measured on one epoch of batch_1: 355 of 355 tensors
        reaching ``get_weights`` were exactly fp16-representable against 58 of 355 in
        the live model, which is the ``.half().float()`` signature, and every BN
        ``num_batches_tracked`` arrived as 0 because ``ModelEMA.update`` lerps only
        floating-point tensors.

        So three things rode on that rebind: the fp16 round-trip, an EMA nobody asked
        for whose ``updates`` counter restarts at 0 every round, and -- at
        ``local_epochs > 1`` -- ``best.pt``'s own choice of epoch, scored on this
        vehicle's 280-image val split. Reading the trainer's module removes all three
        at the send site and changes nothing about what is received.

        Falls back to ``model`` before any ``train()`` has run, and says so: a silent
        fallback here would look exactly like the fix working.
        """
        trainer = getattr(self.yolo, "trainer", None) if self.yolo is not None else None
        module = getattr(trainer, "model", None)
        if module is None:
            logger.warning(
                "[Client] No trainer module available; sending yolo.model instead. "
                "After a train() that is the reloaded fp16 EMA of best.pt, not the "
                "weights this round produced.")
            return self.model
        return module

    def _validate_batch_id(self, batch_id: int) -> bool:
        """Validate that batch_id is within the acceptable range."""
        if not isinstance(batch_id, int):
            logger.warning(f"[Client] Batch ID must be an integer. Got {type(batch_id)}.")
            return False
        if batch_id < self.batch_id_range[0] or batch_id > self.batch_id_range[1]:
            logger.warning(f"[Client] Batch ID {batch_id} is out of range {self.batch_id_range}.")
            return False
        return True    

    def _get_data_yaml_path(self, batch_id):
        """
        Return a runtime data.yaml with correct absolute paths for this machine.

        Uses materialize_data_yaml(), which writes a gitignored data.runtime.yaml
        instead of mutating the tracked data.yaml.

        Args:
            batch_id: The batch ID

        Returns:
            Path: Path to the runtime data.yaml (falls back to the tracked one).
        """
        if os.path.exists(get_data_yaml_path(batch_id)):
            return materialize_data_yaml(batch_id)
        return get_data_yaml_path(batch_id)

    def fit(self, ins: FitIns) -> FitRes:
        if self.model is None:
            logger.error("[Client] No model available, cannot train.")
            # Return original parameters so server isn't disrupted
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={"train_loss": float("inf")},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Failed to load model for training"),
            )
            
        # 1) Load weights from parameters 
        weights_list = self._parameters_to_list(ins.parameters)
        if not weights_list:
            logger.warning("[Client] Received empty weights. Using current model weights.")
            weights_list = get_weights(self.model)
        
        # Track weights checksum for debugging weight transfer
        weights_checksum = sum(w.sum() for w in weights_list if w.size > 0)
        logger.info(f"[Client] Received weights with checksum: {weights_checksum}")
        
        try:
            # set_weights returns False on a count/shape mismatch instead of raising.
            # Ignoring it made a failed load look like a successful round: the client
            # trained from its own stale weights and the server averaged them in.
            if not set_weights(self.model, weights_list, keep_local=self._bn_keys):
                raise ValueError("set_weights returned False - mismatch or error.")
            logger.info("[Client] Successfully applied received weights to model")
        except Exception as e:
            logger.error(f"[Client] set_weights failed: {e}", exc_info=True)
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={"train_loss": float("inf")},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Failed to load weights for training"),
            )

        # Snapshot the global weights so we can apply a FedProx proximal pull
        # after local training (see step 6).
        global_weights = [w.copy() for w in weights_list]

        # 2) Parse config from server
        batch_id = ins.config.get("batch_id", None)
        local_epochs = ins.config.get("local_epochs", self.local_epochs)
        proximal_mu = float(ins.config.get("proximal_mu", 0.0))
        # Whether THIS round draws Ultralytics' diagnostic pictures. The server sends
        # True only on the last round -- see fit_config_fn. Everything else here is a
        # per-run knob that the server forwards from run_config unchanged.
        plots = bool(ins.config.get("plots", True))
        optimizer = str(ins.config.get("optimizer", "auto"))
        lr0 = float(ins.config.get("lr0", 0.0))
        mosaic = float(ins.config.get("mosaic", -1.0))   # negative = leave the default
        # How many leading layers to freeze THIS round. The server sends a non-zero
        # value on round 1 only, so the head can settle against features the backbone
        # already had rather than pulling them apart. 0 = train everything.
        freeze = int(ins.config.get("freeze", 0))
        # FixBN: normalise with the aggregate's statistics and stop them moving.
        # The server decides which rounds, because the warm-up phase is the method.
        fix_bn = bool(ins.config.get("fix_bn", False))
        # Input resolution for this round. BDD frames are 1280x720; at 640 the small
        # classes are sub-20px. The holdout and the baseline are scored at the same
        # size by the pipeline, so this does not quietly become an unfair win.
        imgsz = int(ins.config.get("imgsz", DEFAULT_IMAGE_SIZE))

        # `optimizer="auto"` REPLACES lr0 with 0.002*5/(4+nc) and logs that it did, in
        # a line nobody read. Passing lr0 while leaving the optimizer on auto is a
        # silent no-op that looks like a learning-rate experiment -- it is what makes
        # PR #53's anneal result untrustworthy. Refuse to let it happen quietly.
        if lr0 > 0 and optimizer == "auto":
            logger.warning(
                f"[Client] lr0={lr0} was requested but optimizer='auto', which "
                f"discards lr0 and substitutes its own. This round is NOT running at "
                f"the learning rate it was asked for. Set `optimizer` (e.g. 'AdamW' "
                f"or 'SGD') in run_config to make lr0 take effect."
            )

        # Use stored batch_id as fallback if available
        if batch_id is None:
            logger.warning("[Client] No batch_id received, using stored batch_id.")
            batch_id = self.batch_id

        # Validate batch_id
        if batch_id is None or not self._validate_batch_id(batch_id):
            logger.error(f"[Client] Invalid batch_id: {batch_id}. Expected range: {self.batch_id_range}.")
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message=f"Invalid batch_id: {batch_id}"),
            )

        # Store valid batch_id for future use
        self.batch_id = batch_id
        logger.info(f"[Client] Starting local training with batch_id={self.batch_id}, local_epochs={local_epochs}")

        # 3) Build data path and verify it exists
        data_yaml_path = str(self._get_data_yaml_path(self.batch_id))
        if not os.path.exists(data_yaml_path):
            logger.error(f"[Client] Training data.yaml not found: {data_yaml_path}")
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Missing training data.yaml"),
            )

        # 4) Train the model
        try:
            logger.info(f"[Client] Training with data config: {data_yaml_path}")

            # Ultralytics accumulates gradients up to a nominal batch of 64, so it
            # only calls optimizer.step() every round(64/batch) batches. On a small
            # shard an entire round can finish without a single step: training
            # "succeeds", metrics are logged, and the returned weights are bit-for-bit
            # what the server sent. Warn rather than fail — a caller may want this.
            batch = get_optimal_batch_size(imgsz)
            n_train = count_shard_examples(self.batch_id, "train")
            steps = math.ceil(n_train / batch) * int(local_epochs)
            accumulate = max(round(NOMINAL_BATCH_SIZE / batch), 1)
            if steps < accumulate:
                logger.warning(
                    f"[Client] batch={batch} over {n_train} images x {local_epochs} epoch(s) "
                    f"gives {steps} batch(es), fewer than the {accumulate} needed for one "
                    f"optimizer step. This round will not change the weights. Raise "
                    f"local_epochs or lower the batch size."
                )
            # Only what the server actually set travels: passing lr0=0.0 or
            # mosaic=None would override an Ultralytics default with a wrong value
            # rather than leave it alone.
            tuning = {"optimizer": optimizer} if optimizer != "auto" else {}
            if lr0 > 0:
                tuning["lr0"] = lr0
            if mosaic >= 0:
                tuning["mosaic"] = mosaic
            if freeze > 0:
                # Ultralytics freezes `model.0.`..`model.{freeze-1}.` plus `.dfl`
                # (engine/trainer.py:330-352) and raises if nothing is left trainable,
                # so a too-large value fails loudly rather than training nothing. It
                # also puts those layers' BatchNorm into eval() (trainer.py:701-707), so
                # the statistics do not drift either: measured at freeze=10, 0 of 54
                # backbone weights and 0 of 27 backbone running_var changed, against
                # 66/67 and 30/30 in the head. Round 1 therefore returns a bit-identical
                # backbone from every vehicle and federates only the head.
                tuning["freeze"] = freeze
                logger.info(
                    f"[Client] Round {ins.config.get('server_round')}: freezing the "
                    f"first {freeze} layers so the head settles against fixed features.")
            if fix_bn:
                # A trainer subclass, not a call before train(): `_model_train` runs
                # `model.train()` at the start of every epoch, which would undo an
                # eval() set from out here on the second epoch and leave a warm-up
                # phase nobody chose inside the round. See my_project/trainers.py.
                tuning["trainer"] = fixbn_trainer()
                logger.info(
                    f"[Client] Round {ins.config.get('server_round')}: FixBN on -- "
                    f"normalising with the aggregate's statistics, and not moving them.")

            results = self.yolo.train(
                data=data_yaml_path,
                epochs=local_epochs,
                imgsz=imgsz,
                device=self.device,
                batch=batch,
                verbose=False,
                # Ultralytics draws labels.jpg, train_batch*.jpg and, at final_eval,
                # the confusion matrix and PR/F1 curves. Measured at ~1.5x the wall
                # clock of a 1-epoch round on this hardware, per client, per round.
                #
                # Every round wrote them into the SAME directory (exist_ok=True below),
                # so five rounds of six were drawing pictures that the sixth overwrote
                # before anything read them. pipeline/train_artifacts.py serves that
                # directory and its docstring already says it holds only the last
                # round. So this buys the dashboard nothing and costs a third of the
                # run. The server sends plots=True on the final round only.
                plots=plots,
                **tuning,
                # Ultralytics defaults to 8. Inside a Ray actor on Windows that is a
                # deadlock, not a slowdown — spawned dataloader children never join.
                # With workers=0 the JPEG decode runs on the training thread, which is
                # why `cache` exists: it moves the decode out of the step loop instead
                # of trying to move it onto processes that cannot be spawned here.
                workers=0 if IS_WINDOWS else 8,
                cache=self.cache,
                # Without an explicit name every round writes runs/detect/train, train2,
                # ... at ~22 MB each, per client, forever.
                project="runs/fl",
                name=f"batch{self.batch_id}",
                exist_ok=True,
            )
            
            # 5) Process results
            if hasattr(results, "results_dict"):
                # Real shard size → FedAvg aggregation weight proportional to data.
                num_examples = count_shard_examples(self.batch_id, "train")
                results_dict = results.results_dict
                
                # Create metrics dictionary
                metrics = {
                    "precision": results_dict.get("metrics/precision(B)", 0),
                    "recall": results_dict.get("metrics/recall(B)", 0),
                    "mAP50": results_dict.get("metrics/mAP50(B)", 0),
                    "mAP50-95": results_dict.get("metrics/mAP50-95(B)", 0),
                    "fitness": results_dict.get("fitness", 0),
                    "os": OS_NAME  # Include OS information in metrics for tracking
                }
                
                logger.info(f"[Client] {self.batch_id} Training done. metrics={metrics}")
            else:
                logger.warning("[Client] Training did not return metrics.")
                num_examples = 0
                metrics = {"error": "No metrics returned from training", "os": OS_NAME}

            # 6) Extract updated weights for sending back to server.
            # From the TRAINER's module, not from yolo.model -- see trained_model.
            updated_weights = get_weights(self.trained_model)

            # Log both, every round. The two numbers differing is the fix working; them
            # being equal means the trainer was not reachable and the fp16 EMA went out
            # instead, which is the bug this replaced and must never be silent again.
            #
            # Summed straight off the tensors rather than through get_weights: that would
            # copy all 355 of them to host memory as numpy, ~44 MB per client per round,
            # to produce one float. A log line is not worth a second serialisation.
            reloaded_checksum = sum(
                float(t.sum()) for t in self.model.state_dict().values() if t.numel())
            trained_checksum = sum(w.sum() for w in updated_weights if w.size > 0)
            logger.info(
                f"[Client] trained weights checksum {trained_checksum}; the reloaded "
                f"best.pt EMA would have sent {reloaded_checksum}")

            # FedProx: pull the locally-trained weights back toward the global model
            # by factor mu, i.e. w <- w - mu * (w - w_global). mu == 0 is plain FedAvg.
            # This is a round-level approximation of the FedProx proximal term, applied
            # in weight space because Ultralytics' high-level trainer does not expose a
            # per-step loss hook. See docs/FEDPROX.md.
            if proximal_mu > 0 and len(updated_weights) == len(global_weights):
                logger.info(f"[Client] Applying FedProx proximal term (mu={proximal_mu}).")
                updated_weights = [
                    w - proximal_mu * (w - g)
                    for w, g in zip(updated_weights, global_weights)
                ]
                metrics["proximal_mu"] = proximal_mu

            updated_checksum = sum(w.sum() for w in updated_weights if w.size > 0)
            logger.info(f"[Client] Sending back weights with checksum: {updated_checksum}")
            
            return FitRes(
                parameters=self._list_to_parameters(updated_weights),
                num_examples=int(num_examples),
                metrics=metrics,
                status=Status(code=Code.OK, message="Training successful"),
            )
        except Exception as e:
            logger.error(f"[Client] YOLO training failed: {e}", exc_info=True)
            return FitRes(
                parameters=ins.parameters,
                num_examples=1,
                metrics={"error": str(e), "os": OS_NAME},
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="Training process failed"),
            )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        if self.model is None:
            logger.error("[Client] No model available, cannot evaluate.")
            return EvaluateRes(
                loss=float("inf"), 
                num_examples=1, 
                metrics={"os": OS_NAME}, 
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="No model available")
            )
            
        # 1) Load weights from parameters
        weights_list = self._parameters_to_list(ins.parameters)
        if not weights_list:
            logger.warning("[Client] Received empty weights for evaluation. Using current model weights.")
            weights_list = get_weights(self.model)
            
        # Track weights checksum for debugging weight transfer
        weights_checksum = sum(w.sum() for w in weights_list if w.size > 0)
        logger.info(f"[Client] Received evaluation weights with checksum: {weights_checksum}")
            
        # 2) Apply weights to model
        try:
            logger.info("[Client] Setting evaluation weights.")
            # Same filter as fit(). Under FedBN a client evaluating with the averaged
            # BN would be scoring a model it never trains as, and the self-reported
            # metric would understate the method it is meant to measure.
            success = set_weights(self.model, weights_list, keep_local=self._bn_keys)
            if not success:
                raise ValueError("set_weights returned False - mismatch or error.")
        except Exception as e:
            logger.error(f"[Client] evaluate() set_weights failed: {e}", exc_info=True)
            return EvaluateRes(
                loss=float("inf"),
                num_examples=1,
                metrics={"os": OS_NAME},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="Failed to load weights for evaluation"),
            )

        # 3) Parse config from server
        batch_id = ins.config.get("batch_id", None)
        
        # Use stored batch_id as fallback if available
        if batch_id is None:
            logger.warning("[Client] No batch_id received for evaluation, using stored batch_id.")
            batch_id = self.batch_id
            
        # Validate batch_id
        if batch_id is None or not self._validate_batch_id(batch_id):
            logger.error(f"[Client] Invalid batch_id for evaluation: {batch_id}. Expected range: {self.batch_id_range}.")
            return EvaluateRes(
                loss=float("inf"),
                num_examples=1,
                metrics={"os": OS_NAME},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message=f"Invalid batch_id: {batch_id}"),
            )
            
        # Store valid batch_id for future use
        self.batch_id = batch_id
        logger.info(f"[Client] Starting evaluation with batch_id={self.batch_id}")

        # 4) Build data path and verify it exists
        data_yaml_path = str(self._get_data_yaml_path(self.batch_id))
        if not os.path.exists(data_yaml_path):
            logger.error(f"[Client] Evaluation data.yaml not found: {data_yaml_path}")
            return EvaluateRes(
                loss=float("inf"),
                num_examples=1,
                metrics={"os": OS_NAME},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="Missing evaluation data.yaml"),
            )

        # 5) Evaluate the model
        try:
            logger.info(f"[Client] Evaluating with data config: {data_yaml_path}")
            results = self.yolo.val(
                # The same size the round trained at: `on_evaluate_config_fn` is
                # `fit_config_fn`, so this dict already carries it. Scoring at 640 a
                # model trained at 1024 would report a number for neither.
                data=data_yaml_path,
                imgsz=int(ins.config.get("imgsz", DEFAULT_IMAGE_SIZE)),
                device=self.device,
                verbose=False
            )
            
            # 6) Process results
            # Real val-shard size → correct weighting of the aggregated loss.
            num_examples = count_shard_examples(self.batch_id, "val")
            fitness_value = results.results_dict.get("fitness", float("inf"))
            
            # Create metrics dictionary
            metrics = {
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0),
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                "os": OS_NAME  # Include OS information in metrics for tracking
            }

            logger.info(f"[Client] Evaluation done. Loss={fitness_value}, metrics={metrics}, Images Processed={num_examples}")

            return EvaluateRes(
                loss=float(fitness_value),
                num_examples=num_examples,
                metrics=metrics,
                status=Status(code=Code.OK, message="Evaluation successful"),
            )
        except Exception as e:
            error_message = f"YOLO evaluation failed due to: {str(e)}"
            logger.error(f"[Client] {error_message}", exc_info=True)
            return EvaluateRes(
                loss=float("inf"),
                num_examples=1,
                metrics={"error": error_message, "os": OS_NAME},
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message=error_message),
            )

    def _list_to_parameters(self, weights_list) -> Parameters:
        from flwr.common import ndarrays_to_parameters
        return ndarrays_to_parameters(weights_list)

    def _parameters_to_list(self, parameters: Parameters):
        from flwr.common import parameters_to_ndarrays
        return parameters_to_ndarrays(parameters)


def client_fn(context: Context):
    try:
        logger.info("[Client] Creating FlowerClient instance from client_fn.")
        model_path = "models/yolov8s.pt"  # Relative path

        local_epochs = context.run_config.get("local_epochs", 1)
        client_state = context.state
        batch_id_range = context.run_config.get("batch_id_range", DEFAULT_BATCH_ID_RANGE)
        # Read from the run config rather than from a fit instruction: FedAvg shares
        # one FitIns across every client, so anything per-client sent that way arrives
        # as the last value written (the B9 bug). The cache setting is per *run*, so
        # the run config is where it belongs.
        cache = context.run_config.get("cache", "")
        local_bn = bool(context.run_config.get("local_bn", False))

        client = FlowerClient(
            model_path=model_path,
            client_state=client_state,
            local_epochs=local_epochs,
            batch_id_range=batch_id_range,
            cache=cache,
            local_bn=local_bn,
        )
        
        return client
    except Exception as e:
        logger.error(f"[Client] client_fn failed to create FlowerClient: {e}", exc_info=True)
        return None


# So you can start the client with: flwr run --app client:app
app = ClientApp(client_fn=client_fn)
