from typing import List
import warnings
import torch
import numpy as np
import platform
from collections import OrderedDict
# NOTE: this module deliberately does not import ultralytics. The weight-transfer
# helpers (get_weights/set_weights) depend only on torch + numpy, which keeps FedAvg
# serialization testable and importable without the heavy dependency.
from utils.logging_setup import configure_logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = configure_logging("get_set", "logs/get_set.log")

# OS detection
IS_WINDOWS = platform.system() == "Windows"
OS_NAME = platform.system()
logger.info(f"[GetSet] Detected operating system: {OS_NAME}")

# Constants
DEFAULT_NUM_CLASSES = 13  # The default number of classes for our model

# The architecture both server and client must build, so their state_dicts match.
# Ultralytics infers the scale from the *filename*: "yolov8s-13" parses as scale
# 's', matching models/yolov8s.pt. The old models/yolo8n.yaml parsed as 'n' and
# would silently build a nano net to load small weights into.
NUM_CLASSES_MODEL_YAML = "models/yolov8s-13.yaml"

#: BDD100K's 13 classes, in the order the shards' data.yaml declares them. The order
#: IS the class index, so this list and that file must not disagree -- a pipeline test
#: asserts they still match.
BDD_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle",
               "bicycle", "traffic light", "traffic sign", "trailer",
               "other person", "other vehicle"]

#: BDD name -> COCO name where the datasets use different words for a thing a detector
#: sees the same way. Deliberately short: `stop sign` is one *kind* of traffic sign, so
#: this row is an approximation and is marked as one. `rider` is left out on purpose --
#: COCO's nearest label is `person`, and seeding rider from person starts the two
#: classes BDD most often confuses at exactly the same place.
APPROXIMATE_FROM_COCO = {"traffic sign": "stop sign"}

def learning_checksum(arrays) -> float:
    """Sum of the LEARNED tensors only -- the arithmetic the B4 guard runs on.

    Integer buffers are excluded, and that exclusion is the point. `num_batches_tracked` is
    int64 and climbs by one per optimizer step in every BatchNorm layer, so with 57 of them
    and 88 steps a round a naive sum rises about +5 000 every round whatever the weights do.
    A frozen model would still show a moving checksum.

    Both sides use this one function, and that is what makes the round trip checkable: a
    checksum is a SUM, so it is linear, and FedAvg's weighted mean of client weights has to
    produce exactly the weighted mean of their checksums. `pipeline/roundtrip.py` tests that
    identity against the logs. Two different checksum definitions would make the identity
    untestable rather than false, which is worse.
    """
    return float(sum(a.sum() for a in arrays if a.size > 0 and a.dtype.kind == "f"))


def get_normalized_path(path):
    """
    Normalize path to the current operating system format.
    
    Args:
        path: Path to normalize
        
    Returns:
        str: Normalized path for the current OS
    """
    if IS_WINDOWS:
        return str(path).replace('/', '\\')
    else:
        return str(path).replace('\\', '/')

def _detect_head(model):
    """The Detect module, whether given a ``YOLO`` wrapper or a ``DetectionModel``.

    Both spellings are in use here -- the server holds the wrapper, the client passes
    the inner module to set_weights -- and getting this wrong is silent: indexing the
    wrong object raises inside a try/except and warm-starting simply does not happen,
    leaving a random head that looks warmed in every log that does not print a count.
    """
    node = model
    for _ in range(3):
        if isinstance(node, torch.nn.Sequential):
            return node[-1]
        node = getattr(node, "model", None)
        if node is None:
            return None
    return None


def _class_convs(detect):
    """The per-scale 1x1 convolutions whose output channels *are* the classes.

    Found by shape rather than by index. Ultralytics has changed the internals of
    ``cv3`` between minor versions (plain Convs, then DWConv pairs); what has not
    changed is that the branch ends in a Conv2d with one output channel per class.
    """
    convs = []
    for branch in getattr(detect, "cv3", []):
        found = [m for m in branch.modules()
                 if isinstance(m, torch.nn.Conv2d) and m.out_channels == detect.nc]
        if not found:
            return []
        convs.append(found[-1])
    return convs


def warm_start_head(model13, coco_model, names13=None, coco_names=None) -> List[str]:
    """Copy COCO's class rows into the matching rows of the 13-class head.

    ``YOLO(yaml).load(coco.pt)`` transfers 349 of 355 tensors: everything but the
    three classification convolutions, whose shapes cannot match across a different
    class count. Those six tensors are therefore **random**, and round 1 of every
    federation has been spent teaching the head what a car is while backpropagating
    that noise into a backbone that already knew.

    BDD100K and COCO share most of what matters on a road, so the rows for those
    classes are copied instead of drawn from a distribution. Rows with no counterpart
    -- trailer, other person, other vehicle, rider -- keep their random initialisation:
    this warms what it can and lies about nothing.

    Returns the class names it warmed, so the caller can log what actually happened
    rather than that it was attempted. Empty means nothing was copied, which is a
    fallback, not a failure: a shape mismatch means the two heads do not correspond
    and inventing a correspondence would be worse than a random head.
    """
    names13 = names13 or BDD_CLASSES
    try:
        ours, coco = _detect_head(model13), _detect_head(coco_model)
        if ours is None or coco is None:
            logger.warning("[GetSet] head warm start skipped: no Detect head found")
            return []
        our_convs, coco_convs = _class_convs(ours), _class_convs(coco)
        if not our_convs or len(our_convs) != len(coco_convs):
            logger.warning("[GetSet] head warm start skipped: no matching class convs")
            return []

        coco_names = coco_names or getattr(coco_model, "names", None) or {}
        if isinstance(coco_names, dict):
            coco_names = [coco_names[k] for k in sorted(coco_names)]
        index_of = {n: i for i, n in enumerate(coco_names)}

        pairs = []
        for j, name in enumerate(names13):
            source = name if name in index_of else APPROXIMATE_FROM_COCO.get(name)
            if source in index_of:
                pairs.append((j, index_of[source], name))
        if not pairs:
            return []

        with torch.no_grad():
            for our_conv, coco_conv in zip(our_convs, coco_convs):
                # Same input width or the rows are not comparable at all. Checked per
                # scale rather than once: a mismatch here would silently copy garbage.
                if our_conv.weight.shape[1:] != coco_conv.weight.shape[1:]:
                    logger.warning("[GetSet] head warm start skipped: input width differs")
                    return []
                for j, k, _ in pairs:
                    our_conv.weight[j].copy_(coco_conv.weight[k])
                    our_conv.bias[j].copy_(coco_conv.bias[k])

        warmed = [n for _, _, n in pairs]
        logger.info(f"[GetSet] Warm-started {len(warmed)}/{len(names13)} head classes "
                    f"from COCO: {', '.join(warmed)}")
        return warmed
    except Exception as e:
        logger.error(f"[GetSet] warm_start_head error: {e}", exc_info=True)
        return []


def batchnorm_keys(model) -> set:
    """The ``state_dict`` keys that belong to BatchNorm layers.

    Found by module **type**, never by name. Ultralytics names its modules
    positionally (``model.0.bn.weight``), and a substring match on "bn" would both
    miss a renamed layer and catch anything that happens to contain those letters --
    the failure would be a silently partial FedBN, which looks exactly like a working
    one.

    Both parameters (``weight``, ``bias``) and buffers (``running_mean``,
    ``running_var``, ``num_batches_tracked``) are included: FedBN keeps the whole
    normalisation layer local, not only its statistics.
    """
    keys = set()
    try:
        for mod_name, module in model.named_modules():
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue
            prefix = f"{mod_name}." if mod_name else ""
            for name, _ in module.named_parameters(recurse=False):
                keys.add(prefix + name)
            for name, _ in module.named_buffers(recurse=False):
                keys.add(prefix + name)
    except Exception as e:
        logger.error(f"[GetSet] batchnorm_keys failed: {e}", exc_info=True)
        return set()
    return keys


def get_weights(model):
    """
    Extract the FULL model state as a list of NumPy arrays, in ``state_dict`` order.

    Unlike ``model.parameters()`` (learnable tensors only), ``state_dict()`` also
    includes registered buffers — crucially the BatchNorm running statistics
    (``running_mean``, ``running_var``, ``num_batches_tracked``). YOLOv8 is
    BatchNorm-heavy, so transferring only parameters means FedAvg never averages
    those statistics and the federated model is incorrect. We therefore serialize
    the whole ``state_dict``.

    Order: ``state_dict()`` preserves a deterministic insertion order for a fixed
    module structure, so the i-th array here corresponds to the i-th key of
    ``model.state_dict().keys()`` on any process that builds the same architecture.
    Only the ordered arrays travel over the wire; keys are reconstructed locally in
    ``set_weights`` (see its invariant note).

    Args:
        model: A loaded YOLOv8 model instance (DetectionModel / nn.Module).

    Returns:
        A list of NumPy arrays (one per state_dict entry), or empty list on error.
    """
    try:
        logger.debug("[GetSet] Extracting YOLOv8 model state_dict (params + buffers)...")
        weights_list = [t.detach().cpu().numpy() for t in model.state_dict().values()]

        # Calculate checksum for debugging weight transport.
        weights_checksum = sum(w.sum() for w in weights_list if w.size > 0)

        logger.debug(
            f"[GetSet] Extracted {len(weights_list)} state tensors "
            f"(params+buffers) with checksum: {weights_checksum}"
        )
        return weights_list
    except Exception as e:
        logger.error(f"[GetSet] get_weights error: {e}", exc_info=True)
        return []


def set_weights(model, parameters: List[np.ndarray], keep_local: set | None = None) -> bool:
    """
    Apply a full ``state_dict`` (params + buffers) received as ordered NumPy arrays.

    ``keep_local`` -- a set of state_dict keys to leave at the model's CURRENT value
    instead of overwriting from ``parameters``. This is how FedBN is implemented:
    pass ``batchnorm_keys(model)`` and every normalisation layer stays this vehicle's
    own while the rest of the network is the aggregate.

    The arrays for skipped keys still arrive and are still counted -- the positional
    zip against ``state_dict().keys()`` is what makes this whole transfer work, so
    the wire format must not change. They are discarded on receipt, not on send.

    The incoming list is zipped, in order, against ``model.state_dict().keys()``
    recomputed locally, rebuilt into an ``OrderedDict``, and loaded with
    ``load_state_dict(..., strict=True)``.

    **Invariant:** both the server and every client must construct the *identical*
    architecture (``YOLO("models/yolov8s.pt").model`` with ``nc=13``) before calling
    this. Then the local key order matches the order ``get_weights`` used on the
    sender, so positional zip is correct. ``strict=True`` raises loudly if the order
    or set of keys ever drifts — the desired failure mode, not a silent partial load.

    Each integer buffer (notably BatchNorm ``num_batches_tracked``, int64) is cast
    back to the model's own dtype/device for that key, so ``load_state_dict``'s dtype
    check passes (a raw ``from_numpy`` of an int array would otherwise mismatch).

    Args:
        model: The YOLOv8 model instance.
        parameters: Ordered list of NumPy arrays (full state_dict).

    Returns:
        bool: True on success, False on error.
    """
    try:
        logger.debug("[GetSet] Setting YOLOv8 model state_dict (params + buffers)...")

        # Calculate checksum for debugging.
        weights_checksum = sum(w.sum() for w in parameters if w.size > 0)
        logger.debug(f"[GetSet] Applying state with checksum: {weights_checksum}")

        ref_state = model.state_dict()
        keys = list(ref_state.keys())
        if len(parameters) != len(keys):
            logger.error(
                f"[GetSet] State count mismatch: model has {len(keys)} state tensors, "
                f"but received {len(parameters)} arrays. "
                f"(Are server and client building the same architecture?)"
            )
            return False

        keep_local = keep_local or set()
        new_state = OrderedDict()
        kept = 0
        for key, arr in zip(keys, parameters):
            ref = ref_state[key]
            if key in keep_local:
                # FedBN: this layer stays this vehicle's own. Clone, because
                # load_state_dict copies into the live tensors and aliasing the
                # source would make the assignment a no-op that reads as a success.
                new_state[key] = ref.detach().clone()
                kept += 1
                continue
            tensor = torch.as_tensor(arr).to(dtype=ref.dtype, device=ref.device)
            if tensor.shape != ref.shape:
                logger.error(
                    f"[GetSet] Shape mismatch at '{key}': "
                    f"model {tuple(ref.shape)} != received {tuple(tensor.shape)}"
                )
                return False
            new_state[key] = tensor

        model.load_state_dict(new_state, strict=True)
        if kept:
            logger.info(f"[GetSet] FedBN: kept {kept} local tensors, "
                        f"applied {len(keys) - kept} from the aggregate.")
        logger.debug("[GetSet] Model state_dict updated successfully.")
        return True
    except Exception as e:
        logger.error(f"[GetSet] Error in set_weights: {e}", exc_info=True)
        return False
