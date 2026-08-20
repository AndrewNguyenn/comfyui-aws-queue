"""Prompt/workflow extraction helpers — MIRROR of services/status/handler.py.

These functions (model + character/subject derivation) are duplicated here so
the dispatcher can DENORMALIZE model/character/subject onto each job row at write
time (small attrs). That lets the jobs-by-status GSI omit the ~47 KB workflow_json
from its projection — the DynamoDB read-cost fix (see the project memory
'DDB structural fix plan').

SOURCE OF TRUTH: services/status/handler.py — these are a verbatim copy of its
_parse_workflow .. _extract_character block. This file MUST stay in lockstep:
dispatch-time and read-time derivations have to agree or cancel-group's
_group_key matching breaks. Enforced by services/dispatcher/test_extract_lockstep.py
(runs BOTH copies over a real workflow + the booru-prompt corpus and asserts
identical output). If you change the extraction logic in status/handler.py, copy
it here verbatim and re-run that test (see memory 'test extraction against real
workflows').

Why a copy and not a shared import: each Lambda is packaged from its own service
dir via Code.fromAsset (no shared layer), matching the existing flat-module
convention (anima.py, workflow_router.py sit beside the dispatcher handler).
"""
from __future__ import annotations

import json
import re
from typing import Any


def _parse_workflow(workflow_json: str) -> dict:
    """Parse the stored workflow JSON into a node dict, or {} if unusable."""
    if not workflow_json:
        return {}
    try:
        wf = json.loads(workflow_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return wf if isinstance(wf, dict) else {}


def _extract_model(wf: dict) -> str:
    """Best-effort: the primary model a workflow used.

    A workflow loads many models (LoRA, VAE, CLIP, ControlNet…). The main
    model is either a checkpoint (CheckpointLoaderSimple → ckpt_name) or, for
    Flux/Wan-style graphs, a standalone diffusion model (UNETLoader /
    UnetLoaderGGUF / a 'Load Diffusion Model' node → unet_name). We scan for
    either, skipping the auxiliary loaders, and prefer a checkpoint."""
    ckpt = diffusion = None
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = (node.get("class_type") or "").lower()
        inp = node.get("inputs", {}) or {}
        # Skip auxiliary loaders — none of these is "the model".
        if any(x in ct for x in ("lora", "vae", "clip", "controlnet",
                                 "upscale", "ipadapter", "style", "embedding")):
            continue
        if "checkpoint" in ct and ckpt is None:
            ckpt = inp.get("ckpt_name") or inp.get("model_name")
        elif ("unet" in ct or "diffusion" in ct) and diffusion is None:
            diffusion = inp.get("unet_name") or inp.get("model_name") or inp.get("model")
    name = ckpt or diffusion
    if not name or not isinstance(name, str):
        return ""
    # strip directory + extension for a clean display name
    return name.rsplit("/", 1)[-1].rsplit(".", 1)[0]


_PROMPT_MAX = 2000  # cap each prompt — keeps the /jobs list response bounded

# Prompt-specific text input keys — safe to read on any node: a CLIPTextEncode
# (`text`), an SDXL encoder (`text_g`/`text_l`), a String Literal (`string`),
# a wildcard processor (`wildcard`/`populated_text`).
_TEXT_KEYS = ("text", "text_g", "text_l", "string", "wildcard", "populated_text")
# Video families carry the prompt as a DIRECT string on a conditioning node
# rather than through a CLIPTextEncode:
#   MiniMax H3  -> MiniMaxH3ReferenceToVideo.prompt
#   Wan / SCAIL -> WanVideoTextEncodeCached.positive_prompt / .negative_prompt
# `prompt` also lives in _PROVIDER_KEYS, but that tuple is only trusted when a
# node was reached by following a text link — these are reached through
# conditioning, so without listing them here the viewer showed "Not recorded
# for this generation" for every video job.
_DIRECT_PROMPT_KEYS = ("prompt",)
_POLARITY_PROMPT_KEYS = {"positive": "positive_prompt", "negative": "negative_prompt"}
# Conditioning-ish inputs that carry text embeds rather than a CONDITIONING
# type. The upstream walk only follows inputs whose name contains "cond", which
# skipped Wan's `text_embeds` entirely.
_EMBED_KEYS = ("text_embeds",)
# Generic value keys (a primitive's `value`, a passthrough `prompt`). Only
# trusted on a node reached by *following a text link* — i.e. a confirmed
# string-provider — never on an arbitrary conditioning node, where a stray
# string-typed `value` would be mistaken for a prompt.
_PROVIDER_KEYS = ("value", "prompt")
# Concatenation inputs (StringConcatenate: string_a/string_b, …). The Anima
# workflow wires its positive through `StringConcatenate(trigger_words, POSITIVE)`
# then a RegexReplace, so the prompt only surfaces if we follow *both* parts and
# join them — a plain text-link follow dead-ends at the concat node.
_CONCAT_KEYS = ("string_a", "string_b", "string_c", "string_d")


def _generators(wf: dict) -> tuple[list, list]:
    """Split the nodes that actually sample/detail into (samplers, detailers).

    A generator must satisfy *both* signals: a sampler/guider/detailer class
    name AND a positive/negative/conditioning input link. The class check
    rejects routing nodes like rgthree's "Context Big" (which carry
    conditioning links but don't sample); the link check rejects KSamplerSelect
    and SamplerCustomAdvanced (sampler-named but bearing no prompt). Lists keep
    workflow order, so samplers[0] is the primary pass."""
    samplers, detailers = [], []
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        ct = (node.get("class_type") or "").lower()
        is_detailer = "detailer" in ct
        if not (is_detailer or "sampler" in ct or "guider" in ct):
            continue
        inp = node.get("inputs", {}) or {}
        # `text_embeds` qualifies too: Wan/SCAIL samplers take their prompt that
        # way and have no positive/negative/conditioning input at all, so
        # requiring those three made every Wan job look prompt-less.
        if not any(isinstance(inp.get(k), list)
                   for k in ("positive", "negative", "conditioning") + _EMBED_KEYS):
            continue
        (detailers if is_detailer else samplers).append((nid, node))
    return samplers, detailers


def _extract_prompts(wf: dict) -> list[dict]:
    """Every distinct text prompt a workflow used, each with a label.

    Returns a list of {"label": str, "text": str}. A plain txt2img graph
    yields [Positive, Negative]; a graph with detailer nodes (FaceDetailer,
    DetailerForEach, …) carries its own prompts, so each detailer that uses a
    *different* prompt adds its own labelled section. Ordered primary-sampler
    first; identical prompt text is shown only once.

    ComfyUI's API-format graph wires CONDITIONING chains into a sampler's
    `positive` / `negative` inputs (or a guider's `conditioning`); the text
    lives in a CLIPTextEncode `inputs.text`. We walk each conditioning input
    upstream — following only conditioning-named inputs so a ControlNet/Concat
    node can't leak the other polarity's text — until we hit a text string.
    The text input may itself be *wired* from a string-provider node (a
    "String Literal", a wildcard processor), so a text-link is followed too.
    """

    def _node(ref: Any) -> dict | None:
        # A link is [node_id, output_index]; node_id is a str in API format.
        if not (isinstance(ref, list) and ref):
            return None
        n = wf.get(str(ref[0]))
        return n if isinstance(n, dict) else None

    def _resolve(ref: Any, polarity: str, seen: set,
                 depth: int = 0, via_link: bool = False) -> str:
        node = _node(ref)
        if node is None or depth > 16:
            return ""
        nid = str(ref[0])
        if nid in seen:
            return ""
        seen.add(nid)
        inp = node.get("inputs", {}) or {}
        # A direct string in a text input. The generic value/prompt keys count
        # only when this node was reached by following a text link (via_link)
        # — i.e. it is a confirmed string provider — so a stray string-typed
        # `value` on an unrelated node isn't mistaken for a prompt.
        # Polarity-specific direct prompts first, so a node carrying BOTH
        # positive_prompt and negative_prompt returns the half we asked for
        # instead of whichever key happens to come first.
        pol_key = _POLARITY_PROMPT_KEYS.get(polarity)
        if pol_key:
            val = inp.get(pol_key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        keys = _TEXT_KEYS + _DIRECT_PROMPT_KEYS
        if via_link:
            keys = keys + _PROVIDER_KEYS
        for key in keys:
            val = inp.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # A text input wired from a provider node — follow that link. Only the
        # fixed text-content keys are followed (none is a polarity-named input),
        # so this branch structurally cannot cross from positive to negative.
        for key in _TEXT_KEYS:
            got = _resolve(inp.get(key), polarity, set(seen), depth + 1, via_link=True)
            if got:
                return got
        # A string-concatenation node (StringConcatenate) builds the text from
        # several parts — resolve every part and join, so a prompt wired through
        # `concat(trigger_words, prompt)` isn't dropped. Joined with ", " (the
        # exact delimiter doesn't matter for a human-readable display).
        parts = [g for key in _CONCAT_KEYS
                 if (g := _resolve(inp.get(key), polarity, set(seen), depth + 1, via_link=True))]
        if parts:
            return ", ".join(parts)
        # Otherwise walk upstream, but only through conditioning-named inputs so
        # a ControlNetApplyAdvanced (has both `positive` and `negative`) can't
        # cross polarities. `seen` is copied per branch so it guards against
        # path cycles without letting one branch starve a sibling that shares
        # an upstream node (diamond-shaped conditioning graphs).
        for key, val in inp.items():
            if ("cond" not in key.lower() and key != polarity
                    and key not in _EMBED_KEYS):
                continue
            got = _resolve(val, polarity, set(seen), depth + 1)
            if got:
                return got
        return ""

    def _title(node: dict, nid: str) -> str:
        meta = node.get("_meta") or {}
        t = meta.get("title")
        if isinstance(t, str) and t.strip():
            return t.strip()
        return node.get("class_type") or f"node {nid}"

    def _cap(text: str) -> str:
        return (text or "").strip()[:_PROMPT_MAX]

    # Each generator (sampler / guider / detailer) contributes a positive +
    # negative pair. Dedupe at the *pair* level: a detailer whose whole prompt
    # set matches an earlier generator's is dropped, but a detailer with any
    # distinct prompt keeps its full pair — so a section never ends up orphaned
    # from its other half.
    samplers, detailers = _generators(wf)
    sections: list[dict] = []
    seen_pairs: set = set()
    primary_id = samplers[0][0] if samplers else None
    for nid, node in samplers + detailers:
        inp = node.get("inputs", {}) or {}
        pos = _cap(_resolve(
            inp.get("positive") or inp.get("conditioning") or inp.get("text_embeds"),
            "positive", set()))
        neg = _cap(_resolve(inp.get("negative") or inp.get("text_embeds"),
                            "negative", set()))
        if (not pos and not neg) or (pos, neg) in seen_pairs:
            continue
        seen_pairs.add((pos, neg))
        if nid == primary_id:
            p_label, n_label = "Positive", "Negative"
        else:
            title = _title(node, nid)
            p_label, n_label = f"{title} · Positive", f"{title} · Negative"
        if pos:
            sections.append({"label": p_label, "text": pos})
        if neg:
            sections.append({"label": n_label, "text": neg})

    # Fallback: no generator resolved — read CLIPTextEncode nodes in graph
    # order (first = positive, second = negative, rest numbered).
    if not sections:
        seen: set = set()
        for node in wf.values():
            if not isinstance(node, dict):
                continue
            if "cliptextencode" not in (node.get("class_type") or "").lower():
                continue
            t = (node.get("inputs", {}) or {}).get("text")
            if not (isinstance(t, str) and t.strip()):
                continue
            t = _cap(t)
            if t in seen:
                continue
            seen.add(t)
            i = len(sections)
            sections.append({
                "label": "Positive" if i == 0 else "Negative" if i == 1 else f"Prompt {i + 1}",
                "text": t,
            })

    return sections


# Booru-tag prompts lead with quality/score/meta tags and framing/angle/focus
# tags, then the *character* tag — which is itself followed by the series tag.
# So the character is the first tag that isn't one of these leads. This beats a
# "first parenthetical tag" rule: a series like overlord_(maruyama) carries the
# parens, not the character (narberal_gamma), so that rule would pick the series.
_NON_CHARACTER_TAGS = frozenset({
    # quality / aesthetic / meta — Pony/Illustrious prompts (catponyDark is that
    # lineage) almost always lead with these, so they MUST be skipped or every
    # batch collapses under "masterpiece"/"score_9". (score_N is also caught by
    # _SCORE_RE below, which covers score_8_up / score_9_up / … variants.)
    "masterpiece", "masterwork", "best_quality", "high_quality", "normal_quality",
    "low_quality", "worst_quality", "amazing_quality", "great_quality",
    "good_quality", "ultra-detailed", "ultra_detailed", "highly_detailed",
    "very_aesthetic", "aesthetic", "absurdres", "highres", "hires", "lowres",
    "newest", "recent", "oldest", "early", "mid",
    "source_anime", "source_cartoon", "source_furry", "official_art",
    "rating_safe", "rating_questionable", "rating_explicit",
    # NoobAI/Illustrious aesthetic-rating vocabulary
    "very_awa", "worst_aesthetic", "bad_aesthetic", "displeasing",
    "very_displeasing", "detailed", "very_detailed",
    # angle / viewpoint
    "from_front", "from_behind", "from_back", "from_above", "from_below",
    "from_side", "side_view", "back_view", "front_view", "profile",
    "dutch_angle", "low_angle", "high_angle", "overhead_shot", "straight-on",
    "looking_at_viewer", "looking_down", "looking_up", "looking_back",
    "pov", "fisheye", "panorama", "facing_viewer", "facing_away",
    "three-quarter_angle", "three-quarter_view", "aerial_view",
    "bird's-eye_view", "worm's-eye_view",
    # shot / framing
    "close-up", "closeup", "extreme_close-up", "portrait", "upper_body",
    "lower_body", "full_body", "cowboy_shot", "bust_shot", "bust",
    "wide_shot", "medium_shot", "long_shot", "headshot", "cropped",
    "feet_out_of_frame", "establishing_shot", "full_shot", "medium_full_shot",
    # focus
    "face_focus", "breast_focus", "ass_focus", "hip_focus", "foot_focus",
    "eye_focus", "solo_focus", "butt_focus", "thigh_focus", "leg_focus",
    # subject count — never a character name, but sometimes leads
    "solo", "1girl", "2girls", "3girls", "4girls", "multiple_girls",
    "1boy", "2boys", "multiple_boys", "duo", "trio", "group",
    # persona / age / build descriptors — describe the subject, don't name them,
    # and commonly sit between the framing tags and the character (e.g.
    # "...upper_body, mature_female, bronya_rand, honkai:_star_rail, ...").
    "mature_female", "mature_male", "milf", "dilf", "adult", "aged_up",
    "mature", "old_woman", "old_man", "teenage", "teenager", "young_adult",
    # composition / photography leads occasionally seen first — never a
    # character (macro/bokeh observed leading real catpony prompts).
    "dramatic", "cinematic", "dynamic_angle", "dynamic_pose",
    "macro", "bokeh", "depth_of_field", "motion_blur", "chromatic_aberration",
    "lens_flare", "vignetting", "blurry", "blurry_background",
})
# score_9, score_8_up, score_9_up, score_4, … — the whole Pony score ladder.
_SCORE_RE = re.compile(r"^score_\d")

# Appearance / body descriptors. A real-character prompt leads with the
# character (right after the framing tags); some prompts instead describe an
# UNNAMED figure purely by appearance ("...bust_shot BREAK black_hair,
# very_long_hair, pale_skin, 1boy ..."). So if the first non-lead tag is one of
# these, the prompt names no character — we return "" rather than mislabel the
# row with a hair colour. Hair/eyes/skin are matched by suffix (*_hair etc.);
# this set covers the rest (skin tone, build, bust, common hairstyles).
_APPEARANCE_TAGS = frozenset({
    "dark-skinned_female", "dark-skinned_male", "pale", "tan", "tanned",
    "large_breasts", "huge_breasts", "gigantic_breasts", "medium_breasts",
    "small_breasts", "flat_chest", "large_ass", "huge_ass",
    "curvy", "petite", "muscular", "slim", "plump", "thick_thighs",
    "wide_hips", "abs", "freckles", "ponytail", "twintails", "braid",
    "bangs", "ahoge", "bob_cut", "hair_bun",
})


# Booru `_(X)` disambiguator qualifiers that are CATEGORIES, not franchises:
# painting_(object), sword_(weapon), apple_(fruit). A name_(X) tag is only a
# character when X is a franchise — so we skip the ones whose X is a generic
# category. (This is the small, bounded inverse of "X is a franchise"; a real
# character's franchise like one_punch_man is NOT in here.)
_PAREN_QUALIFIERS = frozenset({
    "object", "medium", "artwork", "traditional_media", "weapon", "food",
    "fruit", "vegetable", "vehicle", "animal", "plant", "instrument",
    "furniture", "clothing", "company", "store", "song", "album", "meme",
    "concept", "material", "body_part", "flower",
})


def _paren_content(tag: str) -> str:
    """The X in a name_(X) tag — painting_(object) → 'object'. "" if no parens."""
    i = tag.find("_(")
    return tag[i + 2:-1] if (i >= 0 and tag.endswith(")")) else ""


def _is_lead_tag(norm: str) -> bool:
    """True for tags that precede the character — quality/score/framing/focus/
    count/persona leads, inline <lora:…> tokens, and name_(category) qualifier
    tags (painting_(object)). These are skipped."""
    return (norm in _NON_CHARACTER_TAGS
            or norm.endswith("_focus")     # *_focus is always framing
            or norm.startswith("<")        # <lora:foo:0.8> etc.
            or _paren_content(norm) in _PAREN_QUALIFIERS  # painting_(object)
            or bool(_SCORE_RE.match(norm)))


def _is_appearance(norm: str) -> bool:
    """True for hair/eyes/skin/body descriptors — see _APPEARANCE_TAGS."""
    return (norm.endswith("_hair") or norm.endswith("_eyes")
            or norm.endswith("_skin") or norm in _APPEARANCE_TAGS)


# Known franchise / copyright tags. The character is the tag immediately BEFORE
# the series tag, so recognizing the series is what pins the character (and lets
# any number of descriptors — clothing, persona, appearance — sit in front of it
# without being mistaken for the character). Punctuated series (honkai:_star_rail,
# fate/stay_night) are detected structurally; the plain-named ones below can't
# be, so they're listed. SEED FROM THE PROMPTS — add a line when a brand-new
# franchise first shows up (until then its character falls back to no-character).
_FRANCHISES = frozenset({
    "honkai:_star_rail", "honkai_star_rail", "star_rail", "honkai_impact",
    "honkai_impact_3rd", "genshin_impact", "wuthering_waves", "zenless_zone_zero",
    "marvel_rivals", "marvel", "overwatch", "league_of_legends", "valorant",
    "teen_titans", "dc", "dc_comics", "marvel_comics",
    "nikke", "nikke_goddess_of_victory", "goddess_of_victory:_nikke",
    "stellar_blade", "blue_archive", "azur_lane", "arknights", "fate",
    "fate/stay_night", "fate/grand_order", "fire_emblem", "pokemon",
    "jujutsu_kaisen", "black_clover", "overlord", "overlord_(maruyama)",
    "danmachi", "dungeon_ni_deai_wo_motomeru_no_wa_machigatteiru_darou_ka",
    "tensei_shitara_slime_datta_ken", "naruto", "bleach", "one_piece",
    "spy_x_family", "chainsaw_man", "my_hero_academia", "boku_no_hero_academia",
    "re:zero", "enen_no_shouboutai", "fire_force", "akame_ga_kill",
    "kill_la_kill", "goblin_slayer!", "goblin_slayer", "soul_calibur",
    "nier:automata", "nier_automata", "one_punch_man", "final_fantasy",
    "final_fantasy_vii", "bleach", "dragon_ball", "high_school_dxd",
    "highschool_dxd",
})


def _series_base(tag: str) -> str:
    """The token before a trailing _(...) disambiguator — overlord_(maruyama) →
    'overlord', jingliu_(honkai:_star_rail) → 'jingliu'. Whole tag if no paren."""
    i = tag.find("_(")
    return tag[:i] if i > 0 else tag


def _is_series(tag: str) -> bool:
    """A copyright/franchise tag. name_(series) CHARACTER tags are NOT series —
    their base (the name) isn't a franchise, and the series lives inside the
    parens, so we ignore paren contents (jingliu_(honkai:_star_rail) → not a
    series, even though the parens contain one)."""
    if tag in _FRANCHISES or _series_base(tag) in _FRANCHISES:
        return True
    if "_(" in tag:               # a name_(X) tag whose base isn't a franchise
        return False              # → it's a character, not a series
    # Punctuated series: a slash (fate/stay_night) or an INTERNAL colon
    # (honkai:_star_rail). The colon must be internal so emoji-mouth tags —
    # both :d / :3 (leading) and d: (trailing) — aren't mistaken for a series.
    return "/" in tag or ":" in tag[1:-1]


def _is_paren_character(tag: str) -> bool:
    """A name_(franchise) character tag — has the disambiguator parens and a
    base that is NOT itself a franchise (so overlord_(maruyama) is excluded)."""
    return ("_(" in tag and tag.endswith(")")
            and _series_base(tag) not in _FRANCHISES)


def _norm_tag(t: str) -> str:
    """Normalize a single booru tag for stoplist comparison / display: strip
    emphasis wrapping and a trailing :weight, lowercase.

    Emphasis syntax wraps the WHOLE tag — (tag), ((tag)), [tag], (tag:1.3) — so
    we only unwrap when the tag starts with a bracket. A character tag with
    internal parens, e.g. ais_wallenstein_(danmachi), does NOT start with one,
    so it is left intact (a naive strip(')') would lop off its closing paren)."""
    t = t.strip()
    while t and t[0] in "([{":
        t = t[1:]
        if t and t[-1] in ")]}":
            t = t[:-1]
        t = t.strip()
    if ":" in t:  # (tag:1.3) / tag::1.3 weight — drop a numeric weight tail.
        head, _, tail = t.rpartition(":")
        # The tail may carry a leaked group-close bracket from a multi-tag
        # weighted group — "(a,_b,_c:1.2)" comma-splits to ".._c:1.2)" — strip
        # those before the numeric check so the weight is still recognized.
        tail = tail.strip().rstrip(")]}").strip()
        if head and tail.replace(".", "", 1).isdigit():
            t = head.strip().rstrip(":").strip()  # rstrip handles tag::1.3
    # Stray leading/trailing underscores: some prompts join quality tags as
    # ",_best_quality,_very_aesthetic" so a tag arrives as "_best_quality".
    # Internal underscores (narberal_gamma) are untouched.
    t = t.strip("_").strip()
    return t.lower()


def _character_and_subject(text: str) -> tuple[str, str]:
    """(character, subject) for a booru-style positive prompt.

    The structural invariant booru prompts follow: the character tag sits
    immediately BEFORE the series/copyright tag. Anchoring on the series is what
    makes this robust — any number of leads (quality, framing) AND descriptors
    (clothing, persona, appearance) can precede the character without being
    mistaken for it. So:
      1. character = the (non-lead) tag right before the first series tag;
      2. else a standalone name_(franchise) parenthetical is still a character;
      3. else (no series recognized) fall back to the FIRST content tag — an
         appearance descriptor means an unnamed figure (return it as a `subject`
         hint, e.g. "black_hair" → the viewer shows "black_hair figure"),
         anything else is taken as the character. This recovers a character
         whose franchise isn't in the dict yet but who LEADS the prompt
         (e.g. "princess_hibana, enen_no_shouboutai, ...").
    All empty for a scenery-only prompt.

    The series anchor (1) is what lets arbitrary descriptors (clothing/persona/
    appearance) precede the character; the fallback (3) is what keeps a leading
    character from being lost when its series isn't recognized. The only gap is
    a descriptor-led prompt whose franchise is ALSO unrecognized — rare, and it
    self-heals by adding the franchise to _FRANCHISES."""
    if not text:
        return "", ""
    content = []
    for chunk in text.replace("\n", " ").split(","):
        for piece in chunk.split("BREAK"):
            norm = _norm_tag(piece)
            if norm and not _is_lead_tag(norm):
                content.append(norm)
    # 1. The character is the content tag immediately before the first series.
    for i, tag in enumerate(content):
        if _is_series(tag):
            return (content[i - 1], "") if i > 0 else ("", "")
    # 2. No series anchor — a name_(franchise) parenthetical is still a character
    #    (covers a character whose series tag was omitted from the prompt).
    for tag in content:
        if _is_paren_character(tag):
            return tag, ""
    # 3. No series recognized — fall back to the leading content tag.
    if content:
        first = content[0]
        return ("", first) if _is_appearance(first) else (first, "")
    return "", ""


def _character_from_prompt(text: str) -> str:
    """Just the named character (or "") — see _character_and_subject."""
    return _character_and_subject(text)[0]


def _extract_subject(wf: dict) -> tuple[str, str]:
    """(character, subject) from the primary positive prompt, for grouping/
    labeling the pending queue. See _character_and_subject."""
    for sec in _extract_prompts(wf):
        if sec.get("label") == "Positive":
            return _character_and_subject(sec.get("text", ""))
    return "", ""


def _extract_character(wf: dict) -> str:
    """Just the named character (or "") — see _extract_subject."""
    return _extract_subject(wf)[0]
