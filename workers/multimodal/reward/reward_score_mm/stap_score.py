import base64
import io
import json
import os
import re
import threading
import time

import numpy as np

from STAP.workers.multimodal.reward.reward_score_mm.stap_format_score import format_reward

W_RQ, W_CLIP, W_HPS = 0.60, 0.20, 0.20
W_LAYERS = ["W0", "W1", "W2", "W3", "W4"]
W_PRIMARY, W_OTHER = 0.6, 0.1
N_JUDGE_FRAMES = int(os.environ.get("STAP_JUDGE_FRAMES", "8"))
_VLM_LAST_USAGE = {}
_VLM_LAST_RAW_RESPONSE = ""

LAYER_EVAL_RUBRICS = {
    "W0": {
        "name": "Perceptual & Spatiotemporal Fidelity",
        "dims": [
            ("visual_fidelity",
             "Image fidelity only: no compression artifact, no mosaic / banding / blocking, no extreme blur. "
             "Independent of color/style. Note: frames sampled at even intervals naturally show motion-induced "
             "changes - do NOT confuse temporal change with compression artifact.",
             "10 = all frames crisp, zero artifact | 9 = essentially crisp, one barely-visible artifact | "
             "8 = one frame mild artifact (e.g. slight banding) | 7 = two frames with mild artifact | "
             "6 = mild artifact noticeable on several frames | 5 = persistent mild artifact across most frames | "
             "4 = moderate artifact reducing clarity | 3 = heavy artifact or all frames degraded | "
             "2 = strong blocking/blur on most frames | 1 = severe blocking / mosaic throughout."),
            ("color_palette_match",
             "Color palette matches at the FAMILY level - NOT pixel-exact shade (red includes "
             "crimson/scarlet/maroon/rust/pink-leaning; blue includes navy/cyan/azure/teal; green includes "
             "olive/emerald/lime; etc.). Every named color in its correct FAMILY = 10 regardless of shade. "
             "Only penalize a DIFFERENT family than the prompt named. If the prompt specifies no colors -> N/A.",
             "10 = every named color in correct family across frames | 9 = correct family, slight cross-family tint | "
             "8 = correct family, one color visibly drifted | 7 = one color wrong family on a minor object | "
             "6 = one minor-object color wrong plus slight drift elsewhere | 5 = one color wrong family on a main object | "
             "4 = a main color wrong plus a secondary drift | 3 = multiple colors wrong family | "
             "2 = most colors wrong family | 1 = palette unrelated."),
            ("style_match",
             "Aesthetic style matches at the FAMILY level (painterly / watercolor / sketch / cartoon / "
             "photorealistic / cyberpunk-neon / vintage / minimalist). Belonging to the prompt's named family = 10 "
             "even if it does not pixel-match a specific sub-style (generic painterly oil for 'Van Gogh' still = 10). "
             "If the prompt names no style -> N/A.",
             "10 = clearly the named style family | 9 = mostly the family, slight imperfection | "
             "8 = family with some drift | 7 = broadly the right family | 6 = right family but weakly expressed | "
             "5 = barely recognizable, vaguely on family | 4 = mostly off-family with a hint | 3 = wrong family | "
             "2 = no clear style, faint generic look | 1 = no stylistic intent visible."),
            ("frame_consistency",
             "Pure frame-to-frame stability: no flicker, no color jitter, no sudden lighting jumps. Independent of "
             "motion content. Evenly-sampled frames naturally look different - do NOT confuse temporal motion with "
             "flicker. Flicker = unnatural sudden brightness/color/lighting shifts.",
             "10 = perfectly stable, no flicker | 9 = essentially stable, one near-imperceptible flicker | "
             "8 = one mild flicker between adjacent frames | 7 = two frames with mild flicker | "
             "6 = mild flicker on several transitions | 5 = persistent mild flicker | "
             "4 = mild-to-moderate flicker throughout | 3 = persistent moderate flicker | "
             "2 = strong flicker on most frames | 1 = severe flicker / strobing."),
        ],
        "anchors": [
            ("10", "foggy forest at dawn: all frames crisp, painterly oil texture clear, warm gold light, no flicker",
             "visual_fidelity=10, color_palette_match=10, style_match=10, frame_consistency=10"),
            ("7", "foggy forest, mostly clean but one frame has mild compression banding in the fog",
             "visual_fidelity=7, color_palette_match=10, style_match=10, frame_consistency=9"),
            ("5", "beach scene, clean and no flicker, but no oil-painting style cue visible (prompt asked oil)",
             "visual_fidelity=8, color_palette_match=6, style_match=4, frame_consistency=8"),
            ("3", "beach scene with heavy blocking in several frames + frame-to-frame color jumps; style absent",
             "visual_fidelity=3, color_palette_match=4, style_match=2, frame_consistency=3"),
            ("1", "blurry pixelated mess, no recognizable scene, heavy flicker, no style",
             "visual_fidelity=1, color_palette_match=1, style_match=1, frame_consistency=1"),
        ],
    },
    "W1": {
        "name": "Object & Relation Structure",
        "dims": [
            ("object_presence",
             "All named objects from the prompt are visibly present in at least one frame AND recognizably "
             "rendered (not abstract blobs). This dim gates the others - if presence is low, others reflect that.",
             "10 = every named object present in EVERY sampled frame, each rendered sharply, unambiguously and "
             "flawlessly - zero defect on any object | "
             "9 = all present and clearly recognizable in all frames, all sharp, at most ONE trivial cosmetic flaw | "
             "8 = all present and recognizable, clearly sharp, identity stable every frame, ONE mild imperfection | "
             "7 = all present and recognizable AND you can EXPLICITLY confirm each one is clearly sharp across the "
             "frames | "
             "6 = all present and recognizable, verified across frames, with mild generation softness | "
             "5 = all named objects present and recognizable but with the usual generation softness, OR one not in "
             "every frame (the common 'objects are simply there' case) | "
             "4 = one object ambiguous, only briefly visible, or noticeably imperfect | "
             "3 = ~50% of objects present, or only the main subject is clear | 2 = main subject barely present | "
             "1 = no named object visible."),
            ("count_correctness",
             "ONLY evaluate this dim when the prompt contains an EXPLICIT number or count word (e.g. 'three', "
             "'2', 'a pair of', 'several', 'dozens of'). If the prompt names objects WITHOUT any count word "
             "(e.g. 'a cat', 'apples on a table'), you MUST output \"N/A\" - do NOT infer an implicit count and "
             "do NOT score it (this is the common case). When a count IS specified, it must match EXACTLY.",
             "N/A = NO explicit count word in the prompt (the common case) | "
             "10 = explicit count matched exactly, all instances individually countable | "
             "9 = exact count, one instance slightly ambiguous | 8 = count right but one instance occluded/ambiguous | "
             "7 = off-by-1 | 6 = off-by-1 with ambiguity | 5 = off-by-2 | 4 = off-by-2 to 3 | "
             "3 = off-by-3+ | 2 = far too few/many | 1 = wildly wrong or zero."),
            ("attribute_binding",
             "STRICT cross-binding: each named attribute (color/material/size/shape/pattern) must be on its named "
             "object, not a different one ('red apple on green table' -> apple red AND table green; 'green apple on "
             "red table' is cross-bound -> low). Seeing the right colors somewhere is NOT enough. N/A if no attribute.",
             "10 = every named attribute verifiably on its exact object in EVERY frame, zero ambiguity, perfectly "
             "rendered | "
             "9 = every attribute on the correct object and clear across frames, at most ONE trivially ambiguous "
             "detail | "
             "8 = all attributes on the correct objects, verified, one slightly imperfectly rendered but unambiguous | "
             "7 = all attributes on the correct objects AND each binding EXPLICITLY confirmed frame-by-frame | "
             "6 = all attributes on the correct objects, verified, with one mild drift | "
             "5 = attributes appear on the correct objects but the binding is only loosely confirmable, OR one mild "
             "drift (the common 'attributes look right' case) | "
             "4 = one attribute ambiguous between two objects | 3 = one clear cross-binding swap | "
             "2 = multiple swaps or most attributes wrong/missing | 1 = all wrong/absent."),
            ("spatial_relation",
             "STRICT preposition match: each spatial preposition (above/below/left/right/behind/in-front-of/beside/"
             "on/under/between/inside) must appear as the SAME relation ('book ABOVE laptop' shown as 'book LEFT of "
             "laptop' is a mismatch -> 5, not 10). N/A if the prompt has no spatial preposition.",
             "10 = every named preposition rendered exactly and unambiguously in EVERY frame | "
             "9 = every relation exactly correct and clear across frames, at most ONE trivially imprecise | "
             "8 = all relations correct, verified, one slightly imprecise but unambiguous | "
             "7 = the named relations present AND each EXPLICITLY confirmed (exact preposition, not just 'near') | "
             "6 = all relations present and verified, one loosely placed | "
             "5 = the layout roughly matches but a relation is only loosely placed/approximate (the common 'layout "
             "looks right' case) | "
             "4 = one relation ambiguous or loosely placed | 3 = one clear preposition substitution | "
             "2 = multiple wrong prepositions / most relations wrong | 1 = no spatial structure."),
        ],
        "anchors": [
            ("10", "all 3 red apples visible across frames, exactly 3, all red, all to the left of the blue cup",
             "object_presence=10, count_correctness=10, attribute_binding=10, spatial_relation=10"),
            ("7", "3 red apples but one frame shows only 2 clearly; spatial layout OK",
             "object_presence=9, count_correctness=8, attribute_binding=9, spatial_relation=8"),
            ("5", "only 2 apples instead of 3, but red color + cup-adjacent layout correct",
             "object_presence=7, count_correctness=5, attribute_binding=9, spatial_relation=7"),
            ("3", "1 brown apple (wrong color), no cup, count off-by-2",
             "object_presence=4, count_correctness=3, attribute_binding=3, spatial_relation=2"),
            ("1", "scene shows a chair and a pear, no apples or cup anywhere",
             "object_presence=1, count_correctness=1, attribute_binding=1, spatial_relation=1"),
        ],
    },
    "W2": {
        "name": "Event & Interaction Dynamics",
        "dims": [
            ("action_correctness",
             "The prompt's named action is visibly performed by the named subject (cross-modal verb<->frame match).",
             "10 = the exact named action performed unambiguously across multiple frames, perfectly clear from start "
             "to finish | 9 = the named action clearly performed across frames, at most one frame trivially weak | "
             "8 = action clearly performed, one minor imperfection in execution | "
             "7 = action clearly visible but slightly incomplete or stylized | "
             "6 = the named action is recognizable but only partially executed or briefly shown (the common "
             "'action is happening' case) | 5 = action weakly/ambiguously shown, intent clear but execution poor | "
             "4 = action barely inferable | 3 = only a hint of the action | 2 = wrong-ish or almost absent | "
             "1 = wrong action or none. (>=9 requires naming the frame(s) showing the verb.)"),
            ("motion_continuity",
             "Motion fluid frame-to-frame; velocity plausible. Evenly-sampled frames naturally show gaps - "
             "distinguish real stuttering from sampling-induced apparent jumps.",
             "10 = smooth, velocity coherent across gaps | 9 = smooth, one transition slightly off (likely sampling) | "
             "8 = mostly coherent, one minor jump | 7 = one clear minor velocity jump | "
             "6 = one clear jump plus minor unevenness | 5 = stuttering / unphysical velocity in 1-2 transitions | "
             "4 = stuttering across several transitions | 3 = jerky / teleporting most of clip | "
             "2 = mostly discontinuous | 1 = no continuous motion."),
            ("event_singularity",
             "BOTH must hold for a high score: (a) the named action is visibly performed AND (b) no extra sequential "
             "events. Trivial co-effects (blender spins -> fruit mixes) count as ONE. If the named action is absent, "
             "do NOT score high just because nothing else happened.",
             "10 = the named action performed AND provably nothing else across all frames, perfectly singular | "
             "9 = the named action clear and singular, at most one negligible co-effect | "
             "8 = named action clear with one minor intended side-effect | "
             "7 = named action clear, no extras, but slightly weak | "
             "6 = the named action occurs and roughly one event is shown, but singularity is only loosely "
             "confirmable (the common case) | 5 = partially shown OR one ambiguous extra event | "
             "4 = weak action plus an extra event | 3 = two distinct sequential actions where one was named | "
             "2 = several extra events | 1 = 3+ chained, OR named action absent."),
            ("subject_persistence",
             "The named subject exists in every frame AND is recognizably the SAME entity (identity drift is a "
             "penalty).",
             "10 = the subject is present in EVERY frame and its identity is provably identical throughout (same "
             "instance, no drift) | 9 = present in all frames, identity essentially identical, one trivial "
             "pose/lighting shift | 8 = present in all frames, minor identity drift | "
             "7 = present in all frames but with a noticeable-yet-minor identity/appearance shift | "
             "6 = present across frames but identity coherence is only loosely confirmable, OR absent from one "
             "frame (the common case) | 5 = present in about half the frames | 4 = present in less than half | "
             "3 = in one frame | 2 = barely present once | 1 = absent."),
        ],
        "anchors": [
            ("10", "cat starts, mid-stride across frames, ends further across the field; smooth motion, one cat throughout",
             "action_correctness=10, motion_continuity=10, event_singularity=10, subject_persistence=10"),
            ("7", "cat runs but motion stutters slightly between two frames; still a single cat throughout",
             "action_correctness=9, motion_continuity=7, event_singularity=10, subject_persistence=10"),
            ("5", "cat moves very slowly across the clip, action intent clear but barely a 'run'",
             "action_correctness=5, motion_continuity=5, event_singularity=8, subject_persistence=10"),
            ("3", "cat in first frame, gone in the middle, reappears teleported at the end",
             "action_correctness=4, motion_continuity=2, event_singularity=6, subject_persistence=4"),
            ("1", "no cat anywhere, only a static landscape (prompt was 'cat runs')",
             "action_correctness=1, motion_continuity=1, event_singularity=1, subject_persistence=1"),
        ],
    },
    "W3": {
        "name": "Intuitive Physics & Plausibility",
        "dims": [
            ("physics_law_adherence",
             "Forces / acceleration / friction / momentum behave correctly. Earth gravity is the default unless the "
             "prompt specifies otherwise.",
             "10 = every force interaction (gravity, momentum, friction, acceleration) provably correct in every "
             "frame | 9 = dynamics correct throughout, one negligible imperfection | "
             "8 = dynamics correct with one minor deviation | 7 = mostly correct, a couple of minor deviations | "
             "6 = broadly plausible but one noticeable soft violation | "
             "5 = physics looks roughly plausible but is not clearly verifiable, OR one clear violation in 1-2 "
             "frames (the common 'looks okay-ish' case) | 4 = a clear violation plus minor ones | "
             "3 = multiple violations | 2 = mostly wrong dynamics | "
             "1 = core physics violated (e.g. floating against gravity)."),
            ("material_realism",
             "Materials deform / break / splash / fracture / melt realistically WHEN the event expects it. "
             "N/A if no material response is expected (e.g. a rigid marble rolling on a hard surface).",
             "10 = material response (deform/break/splash/melt) flawless and fully realistic when expected | "
             "9 = realistic, one negligible defect | 8 = realistic with a minor texture/shape defect | "
             "7 = minor shape/texture defect | 6 = noticeable but still plausible defect | "
             "5 = stylized/simplified but recognizable, realism not clearly confirmable (the common case) | "
             "4 = stylization starts breaking plausibility | 3 = stylization breaks plausibility | "
             "2 = largely wrong response | 1 = wrong response (solid passes through solid)."),
            ("scene_stability",
             "Scene props / geometry / surfaces don't pop / teleport / vanish / morph during the event. ALWAYS "
             "APPLICABLE (no N/A): even with no named props, inspect for unintended pops, flickers, topology changes.",
             "10 = every prop/surface/geometry provably stable across all frames, zero pop/flicker/morph | "
             "9 = stable, one barely-noticeable instability you must look for | 8 = mostly stable, one minor flicker | "
             "7 = one prop briefly flickers/shifts | 6 = one flicker plus a minor shift | "
             "5 = broadly stable but minor pops/shifts are present or not fully verifiable (the common case), OR one "
             "prop pops in/out | 4 = a pop plus a shift | 3 = multiple instability events | "
             "2 = frequent pops/warps | 1 = geometry collapses / rebuilds / massive teleport."),
            ("temporal_physics_coherence",
             "Physics behavior is consistent across all frames - no sudden physics violation mid-clip.",
             "10 = physics provably consistent across every frame transition | 9 = consistent, one negligible wobble | "
             "8 = consistent with one weak moment | 7 = one moment of weak coherence | 6 = a couple of weak moments | "
             "5 = roughly consistent but coherence is not clearly verifiable, OR it breaks once mid-clip (the "
             "common case) | 4 = breaks once plus weak moments | 3 = violations between most frame pairs | "
             "2 = mostly incoherent | 1 = completely incoherent / random."),
        ],
        "anchors": [
            ("10", "marble drops onto sand: above, mid-fall with motion blur, impact forming a crater, then resting with "
                   "realistic sand displacement",
             "physics_law_adherence=10, material_realism=10, scene_stability=10, temporal_physics_coherence=10"),
            ("7", "marble drops onto sand; gravity correct, crater forms but slightly geometric in shape",
             "physics_law_adherence=10, material_realism=7, scene_stability=10, temporal_physics_coherence=9"),
            ("5", "marble drops but the crater is missing at the end (ball clips through sand briefly)",
             "physics_law_adherence=6, material_realism=5, scene_stability=5, temporal_physics_coherence=5"),
            ("3", "marble drops but trajectory is wrong (sideways) and sand doesn't deform at all",
             "physics_law_adherence=3, material_realism=3, scene_stability=6, temporal_physics_coherence=3"),
            ("1", "marble floats sideways against gravity, then teleports across screen; sand untouched",
             "physics_law_adherence=1, material_realism=2, scene_stability=1, temporal_physics_coherence=1"),
        ],
    },
    "W4": {
        "name": "Causal & Counterfactual Coherence",
        "dims": [
            ("event_count_match",
             "Number of distinct visible events matches the prompt (typically 2-4 events).",
             "10 = the exact number of distinct events provably shown, each clearly separated | "
             "9 = exact count, one event boundary slightly merged | 8 = count right, one event compressed | "
             "7 = exact count AND each event cited in a frame | 6 = roughly right, one merged/weak | "
             "5 = roughly the right number but boundaries unclear | "
             "4 = the event count is only loosely matched, OR one event missing (the common case for generated "
             "clips) | 3 = only about half the events visible | 2 = most events missing | "
             "1 = number bears no relation to the prompt."),
            ("event_order",
             "Visible events appear in the prompt's order. N/A ONLY if event_count_match = 1 (0-1 visible events). "
             "With >=2 visible events the order IS evaluable even if some events are missing.",
             "10 = the full event order provably matches the prompt, each transition clearly visible across frames | "
             "9 = order matches, one boundary ambiguous | 8 = order correct with minor ambiguity | "
             "7 = order clearly correct for all visible events, each transition cited frame-by-frame | "
             "6 = order plausible with one near-swap | 5 = order broadly plausible but only partially verifiable | "
             "4 = order is NOT clearly provable from the sparse sampled frames (the common case - a few frames "
             "rarely demonstrate full ordering) | 3 = one adjacent pair appears swapped or order doubtful | "
             "2 = mostly out of order | 1 = no recognizable temporal order. NOTE: a handful of sampled frames "
             "usually cannot PROVE ordering - default to 4 unless you can cite each transition."),
            ("causal_dependency",
             "Later events visibly depend on / are caused by earlier ones (cause->effect link visible). N/A ONLY if "
             "event_count_match = 1. With >=2 visible events, judge the causal link.",
             "10 = the full cause->effect chain (A->B->C) provably visible across frames | "
             "9 = chain clear, one link slightly implicit | 8 = chain mostly clear, one minor weak link | "
             "7 = mostly clear, one weak link | 6 = partly clear, links somewhat implicit | "
             "5 = some causal link suggested but not clearly visual | "
             "4 = causation only inferable, not visually demonstrated across the sampled frames (the common case) | "
             "3 = events appear independent | 2 = mostly disconnected | "
             "1 = events contradict expected causation."),
            ("event_completeness",
             "Every event named in the prompt is visibly shown in at least one frame (distinct from count: asks if "
             "the SPECIFIC events occurred).",
             "10 = every named event provably shown in its own dedicated visible frame | "
             "9 = all events visible, one only briefly | 8 = all visible, one marginal | "
             "7 = all named events visible AND each cited in a specific frame | 6 = most events visible, one weak | "
             "5 = most named events visible, one not clearly shown | "
             "4 = only some named events clearly visible; at least one missing or only implied (the common case - "
             "sparse frames miss intermediate events) | "
             "3 = jumps initial -> final, intermediate events missing | 2 = only initial or final shown | "
             "1 = no named event clearly shown."),
        ],
        "anchors": [
            ("10", "chef cracks egg, whisks it in a bowl, pours into a pan, flips the omelet - 4 events, correct order, "
                   "causal chain visible",
             "event_count_match=10, event_order=10, causal_dependency=10, event_completeness=10"),
            ("7", "4 events visible in order, but the causal link between whisking and pouring is weakly shown",
             "event_count_match=10, event_order=10, causal_dependency=7, event_completeness=9"),
            ("5", "chef cracks egg, then jumps to a finished omelet - 2 visible events in correct order, intermediate "
                  "whisking/pouring missing",
             "event_count_match=4, event_order=8, causal_dependency=4, event_completeness=3"),
            ("3", "events in wrong order: omelet appears finished before whisking",
             "event_count_match=8, event_order=2, causal_dependency=3, event_completeness=6"),
            ("1", "only one static scene of the chef holding an egg, no events occur",
             "event_count_match=1, event_order=N/A, causal_dependency=N/A, event_completeness=1"),
        ],
    },
}

LAYER_DIMS = {w: [d[0] for d in LAYER_EVAL_RUBRICS[w]["dims"]] for w in ["W0", "W1", "W2", "W3", "W4"]}


def _norm_layer(p) -> str:
    p = (str(p) if p is not None else "W0").upper().strip()
    return p if p in W_LAYERS else "W0"


def layer_weights(primary) -> dict:
    primary = _norm_layer(primary)
    return {w: (W_PRIMARY if w == primary else W_OTHER) for w in W_LAYERS}


def _decode_frames(mp4_path, n_frames):
    if not mp4_path or not os.path.exists(str(mp4_path)):
        return []
    frames = []
    try:
        import imageio.v2 as imageio
        rdr = imageio.get_reader(str(mp4_path), format="mp4")
        frames = [np.asarray(f) for f in rdr]
        rdr.close()
    except Exception:
        try:
            import cv2
            cap = cv2.VideoCapture(str(mp4_path))
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cap.release()
        except Exception as e:
            print(f"[stap_score] frame decode failed {mp4_path}: {e}")
            return []
    if not frames:
        return []
    arr = np.asarray(frames)
    idx = np.linspace(0, len(arr) - 1, min(n_frames, len(arr))).astype(int)
    return [np.asarray(arr[i]).astype(np.uint8) for i in idx]


_clip = {}
_endpoint_lock = threading.Lock()
_endpoint_next = {"CLIP_SERVER_URLS": 0, "HPS_SERVER_URLS": 0}

def _next_endpoint(list_name, single_name):
    raw = os.environ.get(list_name, "").strip() or os.environ.get(single_name, "").strip()
    urls = [x.strip().rstrip("/") for x in raw.split(",") if x.strip()]
    if not urls:
        return ""
    with _endpoint_lock:
        i = _endpoint_next[list_name] % len(urls)
        _endpoint_next[list_name] += 1
    return urls[i]


def _get_clip():
    if "model" not in _clip:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        path = os.environ.get("CLIP_MODEL_PATH", "").strip()
        if not path or not os.path.isdir(path):
            raise RuntimeError(f"CLIP_MODEL_PATH not a local dir: {path!r} "
                               f"(or set CLIP_SERVER_URL)")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _clip["proc"] = CLIPProcessor.from_pretrained(path, local_files_only=True)
        _clip["model"] = CLIPModel.from_pretrained(path, local_files_only=True).to(dev).eval()
        _clip["dev"] = dev
        print(f"[stap_score] CLIP loaded on {dev} from {path}")
    return _clip["model"], _clip["proc"], _clip["dev"]


def _normalize_clip_t(x: float) -> float:
    return float(max(0.0, min(1.0, (x - 0.15) / 0.25)))


def _clip_frames_server(frames, caption, server):
    import tempfile
    import requests
    from PIL import Image
    scores = []
    for f in frames:
        p = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                Image.fromarray(f).save(tf, format="JPEG")
                p = tf.name
            with open(p, "rb") as fh:
                r = requests.post(
                    f"{server}/clip_t",
                    files={"image": ("frame.jpg", fh, "image/jpeg")},
                    data={"caption": caption},
                    timeout=int(os.environ.get("CLIP_TIMEOUT", 30)),
                    proxies={"http": None, "https": None},
                )
            r.raise_for_status()
            scores.append(float(r.json()["normalized_score"]))
        except Exception as e:
            print(f"[stap_score] CLIP server frame failed: {e}")
        finally:
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_clip_frames(frames, caption) -> float:
    if not frames or not caption:
        return 0.0
    server = os.environ.get("CLIP_SERVER_URLS", "").strip()
    if server:
        scores = [_clip_frames_server([f], caption, _next_endpoint("CLIP_SERVER_URLS", "CLIP_SERVER_URL")) for f in frames]
        return float(sum(scores) / len(scores)) if scores else 0.0
    server = os.environ.get("CLIP_SERVER_URL", "").strip()
    if server:
        return _clip_frames_server(frames, caption, server)
    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        model, proc, dev = _get_clip()
        imgs = [Image.fromarray(f) for f in frames]
        with torch.no_grad():
            inp = proc(text=[caption], images=imgs, return_tensors="pt",
                       padding=True, truncation=True, max_length=77)
            inp = {k: v.to(dev) for k, v in inp.items()}
            out = model(**inp)
            imf = F.normalize(out.image_embeds, dim=-1)
            txf = F.normalize(out.text_embeds, dim=-1)
            raw = (imf * txf).sum(dim=-1).mean().item()
        return _normalize_clip_t(raw)
    except Exception as e:
        print(f"[stap_score] CLIP local failed: {e}")
        return 0.0


_hps = {}


def _get_hps():
    if "model" not in _hps:
        import torch
        import open_clip
        d = os.environ.get("HPS_MODEL_DIR", "").strip()
        wp = os.path.join(d, "HPS_v2.1_compressed.pt")
        if not d or not os.path.isfile(wp):
            raise RuntimeError(f"HPS weight not found: {wp!r} (or set HPS_SERVER_URL)")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m, _, pre = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
        ckpt = torch.load(wp, map_location="cpu")
        m.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        _hps.update(model=m.to(dev).eval(), pre=pre,
                    tok=open_clip.get_tokenizer("ViT-H-14"), dev=dev)
        print(f"[stap_score] HPS v2.1 loaded on {dev} from {d}")
    return _hps["model"], _hps["pre"], _hps["tok"], _hps["dev"]


def _hps_frames_server(frames, prompt, server):
    import tempfile
    import requests
    from PIL import Image
    scores = []
    for f in frames:
        p = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                Image.fromarray(f).save(tf, format="JPEG")
                p = tf.name
            with open(p, "rb") as fh:
                r = requests.post(
                    f"{server}/score",
                    files={"image": ("frame.jpg", fh, "image/jpeg")},
                    data={"prompt": prompt},
                    timeout=int(os.environ.get("HPS_TIMEOUT", 30)),
                    proxies={"http": None, "https": None},
                )
            r.raise_for_status()
            scores.append(float(r.json()["normalized_score"]))
        except Exception as e:
            print(f"[stap_score] HPS server frame failed: {e}")
        finally:
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_hps_frames(frames, prompt) -> float:
    if not frames or not prompt:
        return 0.0
    server = os.environ.get("HPS_SERVER_URLS", "").strip()
    if server:
        scores = [_hps_frames_server([f], prompt, _next_endpoint("HPS_SERVER_URLS", "HPS_SERVER_URL")) for f in frames]
        return float(sum(scores) / len(scores)) if scores else 0.0
    server = os.environ.get("HPS_SERVER_URL", "").strip()
    if server:
        return _hps_frames_server(frames, prompt, server)
    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
        m, pre, tok, dev = _get_hps()
        imgs = torch.stack([pre(Image.fromarray(f)) for f in frames]).to(dev)
        txt = tok([prompt]).to(dev)
        with torch.no_grad():
            imf = F.normalize(m.encode_image(imgs), dim=-1)
            tf = F.normalize(m.encode_text(txt), dim=-1)
            raw = (imf * tf).sum(dim=-1).mean().item()
        return float(max(0.0, min(1.0, (raw - 0.20) / 0.15)))
    except Exception as e:
        print(f"[stap_score] HPS local failed: {e}")
        return 0.0


def _frames_to_content(frames):
    from PIL import Image
    items = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        items.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return items


def _video_content(video_path):
    b64 = base64.b64encode(open(video_path, "rb").read()).decode("utf-8")
    return [{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}]


_JUDGE_SYSTEM = (
    "You are a strict, evidence-grounded text-to-video evaluator. You grade a generated video on five "
    "cognitive layers (W0 perception, W1 object, W2 event & motion, W3 physics, W4 causality). Each layer "
    "has several orthogonal DIMENSIONS, scored on a 1-10 integer scale (or the string \"N/A\" when a "
    "dimension does not apply).\n"
    "Scoring rules:\n"
    "1. EVIDENCE-ONLY. The generated VIDEO is the ONLY ground truth. The prompt is the TARGET that was "
    "requested, not proof of what was generated - judge what the video actually SHOWS.\n"
    "2. CROSS-MODAL. Compare the video against what the prompt asked (objects, counts, colours, attributes, "
    "spatial relations, actions, order). That comparison is the task, NOT hallucination.\n"
    "3. INDEPENDENT, NO HALO. Score each dimension on its own criterion; a beautiful video is not "
    "automatically high on objects, physics or causality.\n"
    "4. PHYSICS CONTEXT. Assume Earth physics (gravity, momentum, object permanence) UNLESS the prompt "
    "implies another regime (sci-fi / zero-G, cartoon, underwater, magical) - then judge against that regime.\n"
    "5. USE EACH DIM'S LADDER + THE FULL RANGE. Pick the integer whose description best matches the evidence; "
    "2,4,6,8,9 are all valid - do NOT snap to the anchor levels. Anchors illustrate the layer's overall mix; "
    "the per-dim ladder is authoritative when they disagree.\n"
    "6. N/A. If a dimension does not apply (e.g. style with no named style, count with no number specified), "
    "output the string \"N/A\" - do NOT default to 5.\n"
    "7. ANTI-RUBBER-STAMP. 10 means the video fully and strictly satisfies the dim throughout; if you cannot "
    "defend it from the video, score <= 8.\n"
    "Output STRICT JSON only - no markdown, no extra text. Write all text in English."
)


def _layer_block(w: str) -> str:
    r = LAYER_EVAL_RUBRICS[w]
    lines = [f"### {w} - {r['name']}", "Dimensions:"]
    for key, crit, scale in r["dims"]:
        lines.append(f"  - {key}: {crit}")
        lines.append(f"      scale: {scale}")
    lines.append("Anchor examples (level: what the video shows -> per-dim scores):")
    for lvl, vid, sc in r["anchors"]:
        lines.append(f"  [{lvl}] {vid} -> {sc}")
    return "\n".join(lines)


def _schema_block() -> str:
    parts = []
    for w in W_LAYERS:
        inner = ", ".join(f'"{k}": <1-10 or "N/A">' for k in LAYER_DIMS[w])
        parts.append(f'  "{w}": {{{inner}}}')
    return ('{\n  "observations": ["...", "..."],\n'
            + ",\n".join(parts) + "\n}")


def _judge_user(description: str) -> str:
    blocks = "\n\n".join(_layer_block(w) for w in W_LAYERS)
    return (
        "Evaluate the video against the prompt using the per-layer DIMENSION rubrics below.\n\n"
        f'PROMPT (the target requested - NOT proof of the result):\n"{description}"\n\n'
        "MEDIA: you are given the generated VIDEO itself - watch it over time. It is the only ground truth.\n\n"
        f"LAYERS & DIMENSIONS:\n{blocks}\n\n"
        "PROCEDURE:\n"
        "1. Watch the video in temporal order (subject, action, scene, defects) - describe only what is visible.\n"
        "2. For EVERY dimension of EVERY layer, pick a 1-10 integer from that dim's scale (or \"N/A\"), "
        "comparing the video to the prompt.\n\n"
        "OUTPUT - a single JSON object EXACTLY in this shape "
        "(every dimension is an integer 1-10 or the string \"N/A\"):\n"
        f"{_schema_block()}"
    )


_COUNT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|pair|couple|"
    r"triple|quadruple|several|few|many|multiple|numerous|some|hundred|thousand)\b", re.I)


def _has_count(text) -> bool:
    return bool(_COUNT_RE.search(text or ""))


def _parse_dim(raw):
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, (float(raw) - 1.0) / 9.0))
    if isinstance(raw, str):
        s = raw.strip()
        try:
            return max(0.0, min(1.0, (float(s) - 1.0) / 9.0))
        except ValueError:
            return None
    return None


def _vlm_judge(media_content, description):
    api = os.environ.get("STAP_VLM_API_BASE", os.environ.get("VLM_API_BASE", "")).strip()
    model = os.environ.get("STAP_VLM_MODEL", os.environ.get("VLM_MODEL", "Qwen2.5-VL-72B-Instruct")).strip()
    if not api or not model or not media_content:
        print("[stap_score] VLM judge not configured (STAP_VLM_API_BASE/MODEL) -> r_Q=0")
        return {}
    import requests
    user_content = list(media_content) + [{"type": "text", "text": _judge_user(description)}]
    responses_mode = os.environ.get("STAP_VLM_API_MODE", "chat").lower() == "responses"
    if responses_mode:
        response_content = []
        for item in user_content:
            if item.get("type") == "video_url":
                continue
            elif item.get("type") == "image_url":
                response_content.append({"type": "input_image",
                                         "image_url": item["image_url"]["url"]})
            elif item.get("type") == "text":
                response_content.append({"type": "input_text", "text": item["text"]})
        payload = {
            "model": model, "instructions": _JUDGE_SYSTEM,
            "input": [{"role": "user", "content": response_content}],
            "max_output_tokens": int(os.environ.get("STAP_VLM_MAX_TOKENS", 1536)),
            "temperature": 0.0,
        }
        url_suffix = "/responses"
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": int(os.environ.get("STAP_VLM_MAX_TOKENS", 1536)),
            "n": 1, "temperature": 0.0,
            "chat_template_kwargs": {
                "thinking": os.environ.get("STAP_VLM_THINKING", "false").lower() == "true"
            },
        }
        url_suffix = "/v1/chat/completions"
    retries = int(os.environ.get("STAP_VLM_MAX_RETRIES", 3))
    global _VLM_LAST_USAGE, _VLM_LAST_RAW_RESPONSE
    _VLM_LAST_USAGE = {}
    _VLM_LAST_RAW_RESPONSE = ""
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{api.rstrip('/')}{url_suffix}",
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": "Bearer " + os.environ["STAP_VLM_API_KEY"]}
                       if os.environ.get("STAP_VLM_API_KEY") else {}),
                },
                json=payload,
                timeout=int(os.environ.get("STAP_VLM_TIMEOUT", 120)),
                proxies={"http": None, "https": None},
            )
            if r.status_code == 503:
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
            r.raise_for_status()
            response_data = r.json()
            _VLM_LAST_USAGE = response_data.get("usage") or {}
            if responses_mode:
                c = response_data.get("output_text", "")
                if not c:
                    for item in response_data.get("output", []):
                        for part in item.get("content", []):
                            if part.get("type") in ("output_text", "text"):
                                c += part.get("text", "")
            else:
                c = response_data["choices"][0]["message"]["content"]
            _VLM_LAST_RAW_RESPONSE = c[:30000]
            d = json.loads(c[c.find("{"): c.rfind("}") + 1])
            if isinstance(d, dict):
                d = {str(k).strip().upper(): v for k, v in d.items()}
            out = {}
            for w in W_LAYERS:
                layer = d.get(w, {})
                layer = layer if isinstance(layer, dict) else {}
                layer = {str(k).strip().lower(): v for k, v in layer.items()}
                out[w] = {k: _parse_dim(layer.get(k.lower())) for k in LAYER_DIMS[w]}
            if "count_correctness" in out.get("W1", {}) and not _has_count(description):
                out["W1"]["count_correctness"] = None
            return out
        except Exception as e:
            print(f"[stap_score] VLM judge attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    return {}


def vlm_layer_dim_scores(frames, description):
    return _vlm_judge(_frames_to_content(frames) if frames else [], description)


def vlm_layer_dim_scores_video(video_path, description):
    if os.environ.get("STAP_VLM_API_MODE", "chat").lower() == "responses":
        frames = _decode_frames(video_path, N_JUDGE_FRAMES)
        return _vlm_judge(_frames_to_content(frames) if frames else [], description)
    try:
        media = _video_content(video_path)
    except Exception as e:
        print(f"[stap_score] read video failed {video_path}: {e}")
        return {}
    return _vlm_judge(media, description)


def vlm_5layer_scores(frames, description) -> dict:
    perdim = vlm_layer_dim_scores(frames, description)
    if not perdim:
        return {w: 0.0 for w in W_LAYERS}
    out = {}
    for w in W_LAYERS:
        vals = [v for v in perdim[w].values() if v is not None]
        out[w] = float(sum(vals) / len(vals)) if vals else 0.0
    print(f"[stap_score] VLM per-layer (mean of dims) -> { {k: round(v,3) for k,v in out.items()} }")
    return out


def compute_rQ(video_path, description, primary_layer):
    perdim = vlm_layer_dim_scores_video(video_path, description)
    scores = {}
    for w_ in W_LAYERS:
        vals = [v for v in perdim.get(w_, {}).values() if v is not None] if perdim else []
        scores[w_] = float(sum(vals) / len(vals)) if vals else 0.0
    w = layer_weights(primary_layer)
    rq = sum(w[k] * scores.get(k, 0.0) for k in W_LAYERS)
    print(f"[stap_score] r_Q per-layer={ {k: round(v,3) for k,v in scores.items()} } "
          f"primary={_norm_layer(primary_layer)} -> {rq:.4f}")
    return float(rq), scores


def _final_video_path(extra_info):
    per = extra_info.get("video_urls_per_round")
    if isinstance(per, (list, tuple, np.ndarray)) and len(per) > 0:
        for p in reversed(list(per)):
            if p and os.path.exists(str(p)):
                return p
    v = extra_info.get("video_urls")
    if isinstance(v, (list, tuple, np.ndarray)):
        v = v[-1] if len(v) > 0 else None
    if v and os.path.exists(str(v)):
        return v
    return None


def compute_score(prediction: list, ground_truth: list, extra_info=None):
    extra_info = extra_info if isinstance(extra_info, dict) else {}

    assert len(prediction) > 0, "[stap_score] empty model responses!"
    format_score, generate_count = format_reward(prediction)
    if format_score != 1.0:

        reward = -1.0 + format_score
        print(f"[stap_score] FORMAT imperfect (format={format_score:.3f}, "
              f"generate_count={generate_count}) -> reward={reward:.4f} (anchors skipped)")
        extra_info["_stap_metrics"] = {
            "format_score": format_score, "generate_count": generate_count, "gated": True,
            "reward": reward, "layer": _norm_layer(extra_info.get("layer", "W0")),
        }
        return float(reward)

    video_path = _final_video_path(extra_info)
    if not video_path:
        print("[stap_score] format OK but no generated video found -> reward=0.0")
        extra_info["_stap_metrics"] = {
            "format_score": 1.0, "generate_count": generate_count, "no_video": True,
            "reward": 0.0, "layer": _norm_layer(extra_info.get("layer", "W0")),
        }
        return 0.0

    description = ground_truth[0] if ground_truth else ""
    primary = extra_info.get("layer", "W0")
    frames = _decode_frames(video_path, N_JUDGE_FRAMES)
    if not frames:
        print(f"[stap_score] could not decode {video_path} -> reward=0.0")
        extra_info["_stap_metrics"] = {
            "format_score": 1.0, "generate_count": generate_count, "decode_fail": True,
            "reward": 0.0, "layer": _norm_layer(primary),
        }
        return 0.0

    r_q, rq_layers = compute_rQ(video_path, description, primary)
    r_clip = compute_clip_frames(frames, description)
    r_hps = compute_hps_frames(frames, description)
    reward = W_RQ * r_q + W_CLIP * r_clip + W_HPS * r_hps

    extra_info["_stap_metrics"] = {
        "format_score": 1.0, "generate_count": generate_count, "layer": _norm_layer(primary),
        "r_Q": r_q, "r_CLIP": r_clip, "r_HPS": r_hps, "reward": reward,
        "r_Q_per_layer": rq_layers,
    }
    print(
        f"[stap_score] layer={_norm_layer(primary)} | "
        f"r_Q={r_q:.4f}(x{W_RQ}) r_CLIP={r_clip:.4f}(x{W_CLIP}) r_HPS={r_hps:.4f}(x{W_HPS}) "
        f"| video={os.path.basename(str(video_path))} | reward={reward:.4f}"
    )
    return float(reward)

