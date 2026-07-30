#!/usr/bin/env python3
"""Verify pseudo-GT candidates with Qwen3-VL.

Example:
    conda run -n qwen3vl python qwen/verify_pseudo_gt_with_qwen.py \
      --dataset activitynet \
      --resume
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


SUPPORTED_QWEN_MODELS = {
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
    "30b-a3b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
}
DEFAULT_MODEL = "8b"
DEFAULT_MODEL_ID = SUPPORTED_QWEN_MODELS[DEFAULT_MODEL]
DATASETS = ("tvr", "activitynet", "charades", "msrvtt")


def default_input_for_dataset(dataset: str) -> str:
    return f"outputs/upstream/pseudo_gt_candidates.{dataset}.jsonl"


def default_output_for_dataset(dataset: str) -> str:
    return f"outputs/runs/{dataset}/verification.jsonl"


RESULT_DEFAULTS: Dict[str, Any] = {
    "gt_label_recommendation": "reject",
    "reason": "",
    "confidence": 0.0,
}

def make_tvr_qwen_prompt(query: str) -> str:
    return f"""
You are a strict but fair verifier for adding pseudo ground-truth annotations to a TVR/PRVR dataset.

Input:
- Text query
- Candidate frame sequence from a non-GT moment
- Identity reference frame sequence from a known positive moment

Query:
"{query}"

Task:
Decide whether the candidate frames should be labeled as an additional pseudo ground-truth moment for this query.

Core rule:
Accept only if one clear candidate frame or one visually coherent candidate sub-sequence satisfies the full visible meaning of the query.
Do not require the candidate to match the identity reference scene, background, camera angle, clothing, or episode context unless the query explicitly requires it.

Use of identity reference frames:
Use identity reference frames only to help identify named characters in the candidate.
Do not use identity reference frames as evidence for the candidate action, object, posture, location, relation, displayed text, or event.
If something appears only in identity reference frames but not in candidate frames, treat it as missing in the candidate.

Candidate evidence:
Judge only what is visible in the candidate frames.
Do not infer missing action, object, posture, location, relation, gesture, or screen text from the query, show context, or identity reference frames.
When possible, cite the candidate frame index or range supporting the decision.

Before deciding:
Decompose the query into required visible elements.
Every phrase that describes subject, action, object, posture, source, target, location, background, relation, interaction, ownership, gesture, or displayed text must be checked.
Do not ignore prepositional or modifier phrases such as "on", "from", "in", "at", "with", "while", "to", "into", "out of", or "next to".

Required query elements:
- required subject(s), especially named characters,
- exact visible action/event/state,
- required object,
- required posture or body state,
- explicitly mentioned source/target of motion or action,
- explicitly mentioned location/background/place,
- explicitly mentioned relation, interaction, ownership, gesture, or displayed text.

Subject identity:
For named characters, first verify that the candidate contains a person visually consistent with the identity reference character.
Do not assign the query name to a candidate person only because the action, location, or context matches the query.
Face, side view, back view, hairstyle, body shape, clothing, and context may help identity matching.
Reject if the visible candidate person is clearly inconsistent with the identity reference, even if the action/event matches.
Reject only if the required character is absent, clearly different, or genuinely ambiguous.

Action/event:
The query-required subject must perform the query-required visible action/event/state in the candidate.
Reject if a different person performs the action.
Reject if the correct subject is present but performs a different action.
Other people may also act, but the required subject must satisfy the query.

Dynamic actions:
The full start-to-end motion does not need to be visible.
Accept if the candidate shows a clear key phase, contact, or immediate result that directly demonstrates the action.
However, if the query specifies a source, target, posture, or location of the action, that condition must also be visible.
Do not accept a merely related state without the required action cue or object interaction.

Specific gestures or manipulations:
If the query describes a specific gesture, hand motion, or object manipulation, the candidate must show that specific form.
Do not accept a generic or similar gesture when the query requires a more specific one.

Dialogue/audio:
If the query includes speech context that cannot be verified visually, do not reject only for that.
Evaluate the visible requirements: subject, action, object, posture, location, and relation.

Object/location/text/relation:
Required objects must be visible and involved in the action.
Required posture, source, target, and location/background phrases are required when mentioned in the query.
If the query requires text on a screen, display, document, card, sign, or computer, that text must be visible in the candidate.
For multi-subject queries, all required subjects must be identifiable enough and involved with each other.

Decision consistency:
The label, reason, and confidence must agree.
If all required visible elements are satisfied, return accept.
If any required visible element is missing or uncertain, return reject.
The reason must start with "ACCEPT |" for accept and "REJECT |" for reject.

Return only valid JSON:
{{
  "gt_label_recommendation": "accept" or "reject",
  "reason": "ACCEPT | or REJECT | subject=...; identity_evidence=...; action=...; object=...; posture=...; location/background=...; source/target/relation=...; gesture/text=...; candidate_evidence=frame indices or range",
  "confidence": 0.0 to 1.0
}}
""".strip()





def make_activitynet_qwen_prompt(query: str) -> str:
    return f"""
You are a strict verifier for adding pseudo ground-truth annotations to an ActivityNet Captions PRVR dataset.

Input:
- Text query
- GT/reference frames from the known positive moment
- Candidate frames from a non-GT moment

Query:
"{query}"

Task:
Decide whether the candidate frames should be labeled as an additional positive moment for this query.

Use of GT/reference frames:
Use GT/reference frames only to clarify ambiguous visual meaning in the query, such as action, object, scene, subject type, or temporal state.
Do not use GT/reference frames as evidence that the candidate is positive.
Do not transfer hidden information from GT frames to the candidate.
Do not require the candidate to match the GT frames in identity, clothing, camera angle, background, or exact layout.
The final decision must be based only on the candidate frames.

ActivityNet-specific rule:
ActivityNet queries usually describe generic activities, not fixed named characters.
Exact person identity matching across GT and candidate frames is not required.
However, every visually required query element must still be satisfied in the candidate.

Core verification principle:
Break the query into atomic visual requirements:
- subject type
- action or state
- object/tool/body part/animal/vehicle/food, if mentioned
- location or scene, if mentioned
- spatial or temporal relation, if mentioned
- subject-action-object relationship

Accept only if all required elements are visually supported in the candidate frames.

Subject and action rule:
The required action or state must be clearly visible in the candidate frames.
The required action must be performed by the required subject on the required object.
Reject if the subject, action, and object appear separately but are not part of the same event.
Reject if the action is performed by the wrong subject type.
Reject if the object or scene is visible but the queried action is not shown.

Temporal rule:
The candidate must show the queried event within the sampled frames.
Do not accept based only on preparation, aftermath, or a nearby before/after step.
If the query describes a process or transition, the process/transition itself should be visible.
If the query describes a state, a clear resulting state can be accepted only when it directly satisfies the query.

Evidence rule:
Do not accept based only on a similar background, related object, or broad activity category.
Required objects, tools, animals, vehicles, food, body parts, or scene elements must be visible or unmistakably implied.
When the decisive action, object, location, or relationship is ambiguous, choose reject.
Prioritize avoiding false positives over increasing the number of accepted samples.

Subject type rule:
For generic terms such as "person", "someone", or "people", identity can be flexible.
For specific visible types such as "man", "woman", "child", "dog", or "group", the candidate should be compatible unless the type is visually ambiguous.

Before deciding, internally check:
- What exactly must be visible for this query to be true?
- Is the required action directly visible in the candidate?
- Are the required subject and object connected by the action?
- Am I accepting because of visual evidence, or only because the scene looks plausible?

Output only valid JSON:
{{
  "gt_label_recommendation": "accept" or "reject",
  "reason": "short explanation focused on the decisive visual evidence or missing requirement",
  "confidence": number between 0 and 1
}}
""".strip()


def make_msrvtt_qwen_prompt(query: str) -> str:
    return f"""
You are a strict but fair verifier for adding pseudo ground-truth annotations to an MSR-VTT text-to-video retrieval dataset.

Input:

* Text query: Usually a short open-domain video caption.
* Candidate frames: From a candidate video moment.
* Visual reference frames: From a known positive moment.

Query:
"{query}"

Task:
Decide whether the candidate contains visual evidence that satisfies the explicit visual meaning of the query.

1. Visual Reference vs. Candidate Evidence

* Use reference frames ONLY to clarify ambiguous text in the query, such as actions, postures, event types, object categories, scene types, or spatial setups.
* Judge the candidate strictly on its own visible evidence.
* Do not transfer hidden information, background, camera angle, clothing, exact identity, or extra details from the reference to the candidate.
* Do not add new required details from the reference frames.
* Do not mention the reference frames in the final reason.
* If a required element appears only in the reference frames, treat it as missing in the candidate.

2. Query Decomposition & Matching Evidence

* Decompose the query into explicitly required visible elements: subject, action/event/state, object, clothing, substance, scene, location, attribute, text/sign, genre, and spatial relation.
* Strictly observe prepositions such as "in", "on", "from", and "with".
* First search for any candidate frame or short candidate frame range that satisfies the full query.
* Accept if the required query elements are clearly satisfied within that matching frame or range, even if other sampled frames show unrelated actions.
* Reject only if no single frame or frame range contains the required subject, action/event/state, and required object/location/relation together.
* Do not over-interpret broad caption words such as "clip", "video", "compilation", "highlights", "cartoon", "commercial", or "show"; accept visually compatible evidence unless an explicit required element is missing or contradicted.
* If any explicitly required element is missing, contradicted, or genuinely ambiguous in the candidate, choose reject to avoid false positives.

3. Subjects & Identity

* Exact identity matching between reference and candidate frames is not required.
* Only unconstrained human terms such as "a person", "someone", or "people" may be treated as any visually compatible human subject.
* Any stated subject count, type, role, age, gender, species, or composition is a hard visual requirement and must be supported in the same matching frame or range.
* Do not relax stated subject constraints into a generic human subject.
* If the candidate evidence contradicts or does not clearly support a stated subject constraint, choose reject.
* The query-required subject must be the one performing the query-required action/event/state.

4. Actions, Transitions, and States

* The subject must clearly perform the queried action. Accept if the candidate shows a clear key phase, contact, manipulation cue, state change, or immediate resulting state.
* Reject if the candidate only shows preparation, unrelated aftermath, or someone simply standing near the target.

5. Objects & Substances Strictness

* Required objects must be actively involved in the action. Do not accept merely similar objects.
* Substances or container contents must be visible or unmistakably supported by candidate-frame evidence.

Output Instructions:

* Cite specific candidate frame indices or ranges to support your decision.
* The label, reason, and confidence must logically agree.
* If the candidate evidence describes a subject count/type/composition different from the query, the label must be "reject".

Return only valid JSON:
{{
"gt_label_recommendation": "accept" or "reject",
"reason": "ACCEPT | or REJECT | query_subject=...; candidate_subject=...; action/state=...; object_match=exact/compatible/not_required/missing/different; required_object/substance/clothing=...; location/relation=required_and_visible/not_required/missing/different; candidate_evidence=frames X-Y show why the candidate satisfies or fails the query",
"confidence": 0.0 to 1.0
}}
""".strip()



def make_charades_qwen_prompt(query: str) -> str:
    return f"""
You are a strict but fair verifier for adding pseudo ground-truth annotations to a Charades-STA PRVR dataset.

Input:
- Text query: Usually a short indoor daily-activity description.
- Candidate frames: From a candidate video moment.
- Visual reference frames: From a known positive moment.

Query:
"{query}"

Task:
Decide whether the candidate frames satisfy the explicit visual meaning of the query.

1. Visual Reference vs. Candidate Evidence
- Clarification, not evidence: Use reference frames ONLY to clarify ambiguous text in the query (e.g., specific actions, postures, or spatial setups).
- Strict isolation: Judge the candidate strictly on its own visible evidence. Do not transfer hidden information, background, camera angle, clothing, or exact identity from the reference to the candidate.
- If a required element appears only in the reference frames, treat it as missing in the candidate. Do not infer missing elements based on general plausibility or similar indoor scenes.

2. Query Decomposition & Event Connection
- Decompose text: Check every specified subject, action, object, clothing, substance, appliance, room, and spatial relation (strictly observe prepositions like "in", "on", "from", "with").
- Single connected event: All required elements must be visible and interact to form ONE connected event in the candidate. Reject if they appear separately without connection.
- Prioritize precision: If any explicitly required element is missing, contradicted, or genuinely ambiguous in the candidate, choose reject to avoid false positives.

3. Subjects & Identity
- The subject type must be visually compatible. However, for generic subjects (e.g., "a person", "a man"), exact identity matching between reference and candidate frames is not required.

4. Actions, Transitions, and States
- The subject must clearly perform the queried action. Accept if the candidate shows a clear key phase, contact, manipulation cue, state change, or immediate resulting state.
- Reject if the candidate only shows preparation, unrelated aftermath, or someone simply standing near the target.
- Specific manipulations: 
  * "Opening/closing": Must involve a movable door/cabinet/drawer/panel. Touching or cleaning the surface is insufficient.
  * "Dressing/undressing": The garment type must visually match the query.

5. Objects & Substances Strictness
- Strict match: Required objects must be actively involved in the action. Do not accept merely similar objects. 
- Specificity: Generic actions (drinking, eating, holding) are insufficient if the query specifies a more specific object or substance.
- Hidden contents: Substances or container contents must be visible or unmistakably supported by candidate-frame evidence (e.g., liquid color, label, continuous pouring/serving context).

Output Instructions:
- Cite specific candidate frame indices or ranges to support your decision.
- The label, reason, and confidence must logically agree.

Return only valid JSON:
{{
  "gt_label_recommendation": "accept" or "reject",
  "reason": "ACCEPT | or REJECT | subject=...; action/state=...; object_match=exact/compatible/not_required/missing/different; required_object/substance/clothing=...; location/relation=required_and_visible/not_required/missing/different; candidate_evidence=frames X-Y show why the candidate satisfies or fails the query",
  "confidence": 0.0 to 1.0
}}
""".strip()








def make_qwen_prompt(query: str, dataset: str = "tvr") -> str:
    if dataset == "activitynet":
        return make_activitynet_qwen_prompt(query)
    if dataset == "charades":
        return make_charades_qwen_prompt(query)
    if dataset == "msrvtt":
        return make_msrvtt_qwen_prompt(query)
    return make_tvr_qwen_prompt(query)


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                yield line_idx, json.loads(line)


def read_jsonl_valid(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                yield line_idx, json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"warning: skipping invalid JSONL line {line_idx} in {path}: {exc}",
                    file=sys.stderr,
                )


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def output_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return str(record["query_id"]), str(record["candidate_video_id"])


def input_key(row: Dict[str, Any]) -> Tuple[str, str]:
    query_id = row.get("query_key") or row.get("desc_id")
    candidate_video_id = row.get("pseudo_video_id") or row.get("candidate_video_id")
    return str(query_id), str(candidate_video_id)


def load_done_keys(path: Path) -> set[Tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    for _, record in read_jsonl_valid(path):
        if "query_id" in record and "candidate_video_id" in record:
            done.add(output_key(record))
    return done


def count_existing_accepts(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for _, record in read_jsonl_valid(path)
        if record.get("qwen_recommendation") == "accept" or record.get("add_to_extra_gt") is True
    )


def count_pending_inputs(
    path: Path,
    start_index: int,
    limit: Optional[int],
    done_keys: set[Tuple[str, str]],
) -> int:
    pending = 0
    for line_idx, row in read_jsonl(path):
        if line_idx < start_index:
            continue
        if input_key(row) in done_keys:
            continue
        pending += 1
        if limit is not None and pending >= limit:
            break
    return pending


def validate_frame_paths(paths: Sequence[str], role: str) -> None:
    if len(paths) != 8:
        raise ValueError(f"{role} needs exactly 8 frames, got {len(paths)}")
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"{role} missing frame path: {missing[0]}")


def build_messages(row: Dict[str, Any], dataset: str = "tvr") -> List[Dict[str, Any]]:
    gt_paths = list(row["gt_frame_paths"])
    candidate_paths = list(row["pseudo_frame_paths"])
    if dataset == "tvr":
        content: List[Dict[str, str]] = [
            {
                "type": "text",
                "text": make_tvr_qwen_prompt(str(row["query"])),
            },
            {
                "type": "text",
                "text": (
                    "Candidate frames to judge. Use these frames as evidence for action, "
                    "object, location, relation, and displayed text."
                ),
            },
        ]
        for path in candidate_paths:
            content.append({"type": "image", "image": path})
        content.append(
            {
                "type": "text",
                "text": (
                    "Identity reference only. These frames may be used only to identify named characters. "
                    "Do not use them as evidence for action, object, location, relation, or displayed text."
                ),
            }
        )
        for path in gt_paths:
            content.append({"type": "image", "image": path})
        content.append(
            {
                "type": "text",
                "text": "Now decide using candidate-frame evidence only. Return only valid JSON.",
            }
        )
        return [{"role": "user", "content": content}]

    if dataset == "charades":
        content = [
            {
                "type": "text",
                "text": make_charades_qwen_prompt(str(row["query"])),
            },
            {
                "type": "text",
                "text": (
                    "Candidate frames to judge follow in temporal order. "
                    "Use these frames as evidence for the final decision."
                ),
            },
        ]
        for path in candidate_paths:
            content.append({"type": "image", "image": path})
        content.append(
            {
                "type": "text",
                "text": (
                    "Visual reference frames from a known positive moment follow in temporal order. "
                    "Use them only to clarify ambiguous visual meaning in the query."
                ),
            }
        )
        for path in gt_paths:
            content.append({"type": "image", "image": path})
        content.append(
            {
                "type": "text",
                "text": "Now decide using candidate-frame evidence only. Return only valid JSON.",
            }
        )
        return [{"role": "user", "content": content}]

    if dataset == "msrvtt":
        content: List[Dict[str, str]] = [
            {"type": "text", "text": make_msrvtt_qwen_prompt(str(row["query"]))},
            {
                "type": "text",
                "text": (
                    "Candidate frames follow in temporal order. "
                    "Use ONLY these candidate frames as evidence for the final decision. "
                    "When citing evidence, cite Candidate frame N or Candidate frames X-Y."
                ),
            },
        ]
        for idx, path in enumerate(candidate_paths, 1):
            content.append({"type": "text", "text": f"Candidate frame {idx}"})
            content.append({"type": "image", "image": path})
        content.append(
            {
                "type": "text",
                "text": (
                    "Visual reference frames follow in temporal order. "
                    "Use these ONLY to clarify ambiguous query meaning. "
                    "Do NOT cite reference frames as candidate evidence."
                ),
            }
        )
        for idx, path in enumerate(gt_paths, 1):
            content.append({"type": "text", "text": f"Reference frame {idx}"})
            content.append({"type": "image", "image": path})
        content.append(
            {
                "type": "text",
                "text": "Now decide using candidate-frame evidence only. Return only valid JSON.",
            }
        )
        return [{"role": "user", "content": content}]

    content: List[Dict[str, str]] = [
        {
            "type": "text",
            "text": (
                "GT/reference frames follow in temporal order. "
                "Use them only to interpret the query."
            ),
        }
    ]
    for path in gt_paths:
        content.append({"type": "image", "image": path})
    content.append(
        {
            "type": "text",
            "text": (
                "Candidate frames follow in temporal order. "
                "Make the final decision using only these candidate frames."
            ),
        }
    )
    for path in candidate_paths:
        content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": make_qwen_prompt(str(row["query"]), dataset=dataset)})
    return [{"role": "user", "content": content}]


def resolve_model_id(args: argparse.Namespace) -> str:
    if args.model_id:
        return args.model_id
    return SUPPORTED_QWEN_MODELS[args.model]


def resolve_attn_implementation(args: argparse.Namespace, torch_module: Any) -> str:
    if args.attn_implementation != "auto":
        return args.attn_implementation
    if torch_module.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def make_model_kwargs(args: argparse.Namespace, attn_implementation: str) -> Dict[str, Any]:
    return {
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": attn_implementation,
    }


def load_model_and_processor(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_id = resolve_model_id(args)
    processor_kwargs: Dict[str, Any] = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    attn_implementation = resolve_attn_implementation(args, torch)
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            **make_model_kwargs(args, attn_implementation),
        )
    except Exception as exc:
        if args.attn_implementation != "auto" or attn_implementation != "flash_attention_2":
            raise
        print(
            f"flash_attention_2 load failed ({type(exc).__name__}: {exc}); retrying with sdpa",
            file=sys.stderr,
        )
        attn_implementation = "sdpa"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            **make_model_kwargs(args, attn_implementation),
        )

    args.resolved_attn_implementation = attn_implementation
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    return torch, model, processor


def run_qwen(
    row: Dict[str, Any],
    torch_module: Any,
    model: Any,
    processor: Any,
    max_new_tokens: int,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if messages is None:
        messages = build_messages(row)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch_module.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def make_qwen_input_debug_record(
    line_idx: int,
    row: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "input_line_idx": line_idx,
        "query_id": row.get("query_key") or row.get("desc_id"),
        "candidate_video_id": row.get("pseudo_video_id") or row.get("candidate_video_id"),
        "messages": messages,
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def normalize_qwen_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(RESULT_DEFAULTS)
    result.update(parsed)

    result["gt_label_recommendation"] = (
        "accept" if result.get("gt_label_recommendation") == "accept" else "reject"
    )
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    result["confidence"] = max(0.0, min(1.0, confidence))

    if not isinstance(result.get("reason"), str):
        result["reason"] = str(result.get("reason"))
    return result


def invalid_json_result(raw_text: str, error: Exception) -> Dict[str, Any]:
    result = dict(RESULT_DEFAULTS)
    result["reason"] = f"invalid_qwen_json:{type(error).__name__}"
    result["raw_model_output"] = raw_text
    result["parse_error"] = str(error)
    return result


def make_output_record(row: Dict[str, Any], qwen_result: Dict[str, Any]) -> Dict[str, Any]:
    query_id = row.get("query_key") or row.get("desc_id")
    gt_video_id = row.get("original_gt_video_id") or row.get("gt_video_id") or row.get("query_gt_video_id")
    candidate_video_id = row.get("pseudo_video_id") or row.get("candidate_video_id")
    qwen_recommendation = qwen_result.get("gt_label_recommendation")

    return {
        "query_id": query_id,
        "query_key": row.get("query_key"),
        "desc_id": row.get("desc_id"),
        "query": row["query"],
        "gt_video_id": gt_video_id,
        "candidate_video_id": candidate_video_id,
        "gt_frame_paths": row.get("gt_frame_paths"),
        "candidate_frame_paths": row.get("pseudo_frame_paths"),
        "source_candidate": {
            "gt_ts": row.get("gt_ts"),
            "gt_duration": row.get("gt_duration"),
            "gt_frame_indices": row.get("gt_frame_indices"),
            "pseudo_clip_feature_index": row.get("pseudo_clip_feature_index"),
            "pseudo_clip_feature_len": row.get("pseudo_clip_feature_len"),
            "pseudo_center_frame_index": row.get("pseudo_center_frame_index"),
            "pseudo_frame_indices": row.get("pseudo_frame_indices"),
            "model_agreement": row.get("model_agreement"),
        },
        "qwen_result": qwen_result,
        "qwen_recommendation": qwen_recommendation,
        "add_to_extra_gt": qwen_recommendation == "accept",
    }


def summarize_outputs(output_path: Path) -> Dict[str, Any]:
    total = 0
    qwen_counts: Counter[str] = Counter()
    extra_gt_counts: Counter[str] = Counter()
    for _, record in read_jsonl_valid(output_path):
        total += 1
        qwen_counts[str(record.get("qwen_recommendation"))] += 1
        extra_gt_counts[str(record.get("add_to_extra_gt"))] += 1
    return {
        "total": total,
        "qwen_recommendation": dict(qwen_counts),
        "add_to_extra_gt": dict(extra_gt_counts),
    }


def export_subsets(output_path: Path, all_path: Path) -> None:
    def qwen_accepts() -> Iterable[Dict[str, Any]]:
        for _, record in read_jsonl_valid(output_path):
            if record.get("qwen_recommendation") == "accept":
                yield record

    write_jsonl(all_path, qwen_accepts())


def default_summary_path(output_path: Path) -> Path:
    if output_path.name == "verification.jsonl":
        return output_path.with_name("summary.json")
    return output_path.with_suffix(".summary.json")


def default_extra_gt_all_path(output_path: Path) -> Path:
    if output_path.name == "verification.jsonl":
        return output_path.with_name("extra_gt_all.jsonl")
    return output_path.with_name(output_path.stem + ".extra_gt_all.jsonl")


def make_summary(
    total: int,
    qwen_counts: Counter[str],
    extra_gt_counts: Counter[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "total": total,
        "qwen_recommendation": dict(qwen_counts),
        "add_to_extra_gt": dict(extra_gt_counts),
        "model_id": resolve_model_id(args),
        "attn_implementation": getattr(
            args, "resolved_attn_implementation", args.attn_implementation
        ),
    }


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def default_plot_output_dir(output_path: Path) -> Path:
    if output_path.name == "verification.jsonl":
        return output_path.with_name("plots")
    return output_path.with_name(output_path.stem + ".plots")


def render_plots(
    verification_path: Path,
    output_dir: Path,
    decision_filter: Optional[str],
    dpi: int,
    show_filenames: bool,
    overwrite: bool,
    max_pixels: Optional[int],
    backend: str,
) -> int:
    from plot_tvr_qwen_verification import (
        count_jsonl as count_plot_jsonl,
        get_candidate_video_id,
        get_decision,
        get_query_id,
        make_plot_with_backend,
        read_jsonl as read_plot_jsonl,
        should_keep,
        slugify,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    total = count_plot_jsonl(verification_path)
    iterator = tqdm(read_plot_jsonl(verification_path), total=total, desc="plot TVR galleries")
    for record_idx, record in enumerate(iterator):
        if not should_keep(record, decision_filter, None):
            continue

        query_id = slugify(get_query_id(record), max_len=70)
        candidate_id = slugify(get_candidate_video_id(record), max_len=70)
        decision = slugify(get_decision(record), max_len=30)
        plot_path = output_dir / f"{record_idx:06d}_{query_id}_{candidate_id}_{decision}.png"
        if plot_path.exists() and not overwrite:
            continue

        make_plot_with_backend(
            record,
            output_path=plot_path,
            dpi=dpi,
            show_filenames=show_filenames,
            max_pixels=max_pixels,
            backend=backend,
        )
        written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="tvr")
    parser.add_argument("--input", default=None, help="Defaults to outputs/upstream/pseudo_gt_candidates.<dataset>.jsonl")
    parser.add_argument("--output", default=None, help="Defaults to outputs/runs/<dataset>/verification.jsonl")
    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_QWEN_MODELS),
        default=DEFAULT_MODEL,
        help=f"Qwen3-VL preset to load. Default: {DEFAULT_MODEL} ({DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Custom Hugging Face model id. Overrides --model when set.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="auto",
        help="Attention backend. auto prefers flash_attention_2 when available, then falls back to sdpa.",
    )
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-json", action="store_true")
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--extra-gt-all", default=None)
    parser.add_argument("--no-subset-export", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Write summary and accept subset every N newly processed records. Use 0 to disable.",
    )
    parser.add_argument(
        "--print-qwen-input",
        action="store_true",
        help="Print the chat messages sent to Qwen for each verified record to stderr.",
    )
    parser.add_argument(
        "--include-qwen-input",
        action="store_true",
        help="Include the chat messages sent to Qwen in each output JSONL record.",
    )
    parser.add_argument("--plot", action="store_true", help="Render verification frame galleries after Qwen verification.")
    parser.add_argument("--plot-output-dir", default=None)
    parser.add_argument("--plot-decision", choices=["accept", "reject", "none"], default=None)
    parser.add_argument("--plot-dpi", type=int, default=120)
    parser.add_argument(
        "--plot-backend",
        choices=["pil", "matplotlib"],
        default="matplotlib",
        help="Plot renderer. pil is much faster for large ActivityNet frames.",
    )
    parser.add_argument("--plot-show-filenames", action="store_true")
    parser.add_argument("--plot-overwrite", action="store_true")
    args = parser.parse_args()
    if args.input is None:
        args.input = default_input_for_dataset(args.dataset)
    if args.output is None:
        args.output = default_output_for_dataset(args.dataset)
    return args


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and args.overwrite:
        output_path.unlink()
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"Output exists. Use --resume or --overwrite: {output_path}")

    done_keys = load_done_keys(output_path) if args.resume else set()
    mode = "a" if args.resume else "w"

    summary_path = Path(args.summary) if args.summary else default_summary_path(output_path)
    all_path = (
        None
        if args.no_subset_export
        else Path(args.extra_gt_all)
        if args.extra_gt_all
        else default_extra_gt_all_path(output_path)
    )

    pending_total = count_pending_inputs(input_path, args.start_index, args.limit, done_keys)
    existing_summary = summarize_outputs(output_path) if args.resume and output_path.exists() else None
    existing_total = int(existing_summary["total"]) if existing_summary else 0
    existing_accepts = count_existing_accepts(output_path) if args.resume else 0
    qwen_counts = Counter(existing_summary["qwen_recommendation"]) if existing_summary else Counter()
    extra_gt_counts = Counter(existing_summary["add_to_extra_gt"]) if existing_summary else Counter()

    torch_module, model, processor = load_model_and_processor(args)

    processed_new = 0
    accepted_new = 0
    all_f = None
    try:
        if all_path is not None:
            all_path.parent.mkdir(parents=True, exist_ok=True)
            if args.resume and output_path.exists():
                export_subsets(output_path, all_path)
            all_f = all_path.open("a" if args.resume else "w", encoding="utf-8")

        with output_path.open(mode, encoding="utf-8") as out_f, tqdm(
            total=pending_total,
            desc=f"Qwen {args.dataset} verify",
            unit="cand",
            dynamic_ncols=True,
        ) as progress:
            for line_idx, row in read_jsonl(input_path):
                if line_idx < args.start_index:
                    continue
                if args.limit is not None and processed_new >= args.limit:
                    break
                if input_key(row) in done_keys:
                    continue

                try:
                    validate_frame_paths(row["gt_frame_paths"], "gt/reference")
                    validate_frame_paths(row["pseudo_frame_paths"], "candidate")
                except Exception as exc:
                    if not args.skip_missing_images:
                        raise
                    qwen_result = invalid_json_result("", exc)
                    record = make_output_record(row, qwen_result)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    processed_new += 1
                    progress.update(1)
                    qwen_counts[str(record.get("qwen_recommendation"))] += 1
                    extra_gt_counts[str(record.get("add_to_extra_gt"))] += 1
                    if args.checkpoint_every > 0 and processed_new % args.checkpoint_every == 0:
                        out_f.flush()
                        os.fsync(out_f.fileno())
                        if all_f is not None:
                            all_f.flush()
                            os.fsync(all_f.fileno())
                        write_summary(
                            summary_path,
                            make_summary(
                                existing_total + processed_new,
                                qwen_counts,
                                extra_gt_counts,
                                args,
                            ),
                        )
                    progress.set_postfix(
                        {
                            "accept": accepted_new,
                            "qwen": record["qwen_recommendation"],
                            "checkpoint": processed_new,
                        }
                    )
                    continue

                messages = build_messages(row, dataset=args.dataset)
                qwen_input_debug = make_qwen_input_debug_record(line_idx, row, messages)
                if args.print_qwen_input:
                    print(json.dumps(qwen_input_debug, ensure_ascii=False, indent=2), file=sys.stderr)

                raw_output = run_qwen(
                    row=row,
                    torch_module=torch_module,
                    model=model,
                    processor=processor,
                    max_new_tokens=args.max_new_tokens,
                    messages=messages,
                )
                try:
                    qwen_result = normalize_qwen_result(extract_json_object(raw_output))
                except Exception as exc:
                    if args.strict_json:
                        raise ValueError(f"Invalid Qwen JSON at input line {line_idx}: {raw_output}") from exc
                    qwen_result = invalid_json_result(raw_output, exc)

                record = make_output_record(row, qwen_result)
                if args.include_qwen_input:
                    record["qwen_input"] = qwen_input_debug
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                processed_new += 1
                if record["qwen_recommendation"] == "accept":
                    accepted_new += 1

                qwen_counts[str(record.get("qwen_recommendation"))] += 1
                extra_gt_counts[str(record.get("add_to_extra_gt"))] += 1
                if record["qwen_recommendation"] == "accept" and all_f is not None:
                    all_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    all_f.flush()
                if args.checkpoint_every > 0 and processed_new % args.checkpoint_every == 0:
                    out_f.flush()
                    os.fsync(out_f.fileno())
                    if all_f is not None:
                        all_f.flush()
                        os.fsync(all_f.fileno())
                    write_summary(
                        summary_path,
                        make_summary(
                            existing_total + processed_new,
                            qwen_counts,
                            extra_gt_counts,
                            args,
                        ),
                    )

                progress.update(1)
                progress.set_postfix(
                    {
                        "accept": accepted_new,
                        "qwen": record["qwen_recommendation"],
                        "extra_gt": record["add_to_extra_gt"],
                        "checkpoint": processed_new,
                    }
                )

    finally:
        if all_f is not None:
            all_f.close()

    summary = summarize_outputs(output_path)
    summary["model_id"] = resolve_model_id(args)
    summary["attn_implementation"] = getattr(args, "resolved_attn_implementation", args.attn_implementation)
    write_summary(summary_path, summary)
    if all_path is not None:
        export_subsets(output_path, all_path)

    if args.plot:
        plot_output_dir = Path(args.plot_output_dir) if args.plot_output_dir else default_plot_output_dir(output_path)
        plot_count = render_plots(
            verification_path=output_path,
            output_dir=plot_output_dir,
            decision_filter=args.plot_decision,
            dpi=args.plot_dpi,
            show_filenames=args.plot_show_filenames,
            overwrite=args.plot_overwrite or args.overwrite,
            max_pixels=args.max_pixels,
            backend=args.plot_backend,
        )
        summary["plots_written"] = plot_count
        summary["plot_output_dir"] = str(plot_output_dir)
        summary["plot_max_pixels"] = args.max_pixels
        summary["plot_backend"] = args.plot_backend
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
