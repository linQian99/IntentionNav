"""Prompt templates for IntentionNav intent generation via Gemini."""


# Per-category prompt: one Gemini call per (surface, category) manifest entry.
# Step 1: vision-based identification of the SPECIFIC object in the photo
# Step 2: generate intents that don't directly name the object
INTENT_GENERATION_PROMPT = """你是理解人类日常生活场景的专家。请根据照片和场景信息，为指定的目标物品生成隐含意图标注。

## 场景信息
- 房间类型: {room_type}
- 支撑面/参考物: {surface_category} (其上有 {target_category} 类物品)
- 该 surface 上的所有目标类别: {all_categories_on_surface}
- 房间内其他物品: {room_context}
- 物品空间关系: {spatial_relationships}

## 目标类别（本条意图针对）
- 类别: **{target_category}**
- 关系: {target_relation}（on=放在表面上 | above=悬挂在上方）
- 该类共 {target_count} 个实例

## 任务（两步）

### Step 1: 视觉识别（看图）
观察照片中的 {target_category} 类物品，给出**具体描述**：
- 颜色 / 材质 / 形状 / 估算尺寸
- 数量 / 摆放位置（在桌上 / 柜里 / 墙上）
- 状态（开关 / 满空 / 整齐凌乱）
如果无法在图中清楚看到，标注 "not_clearly_visible"。

### Step 2: 生成隐含意图（**恰好 4 条**，不多不少）
基于你识别到的具体物品 + 场景信息，生成 **exactly 4** 条隐含的人类意图描述，每条对应一个不同的使用/需求场景。每条意图需配 4 种风格变体。输出的 `intents` 列表长度必须等于 4；少于或多于 4 条视为错误输出。

## 严格规则（违反则整条作废）

1. **禁止直接命名 `{target_category}` 类别名**（中英所有同义词、部首组合都禁）
2. **禁止使用直接暗示该类的动词**（如 cup→喝/饮，book→读/翻，lamp→开灯）
3. **可以用**：场景（客人、晚餐、纪念日、上床前）、状态（累、渴、无聊、心情好）、
   抽象目标（放松、准备、庆祝、装饰）、氛围（温馨、仪式感、安静）

## 4 种风格变体

- **formal**: 礼貌完整句。例："朋友马上要来做客了，我想好好准备一下。"
- **natural**: 日常自言自语。例："要接待今晚的访客。"
- **casual**: 极简短。例："客人要来。"
- **emotional**: 带情绪/氛围。例："今晚想搞点仪式感。"

## 难度分级
- **easy**: 常识可推（口渴 → 喝水 → cup）
- **medium**: 需场景理解（设宴 → 餐具 → wine_set/plate）
- **hard**: 文化/情境（仪式感 → 装饰摆件 → ornament/vase）

## 输出（严格 JSON）

{{
  "specific_description": {{
    "object": "图中识别到的具体描述（如：白色陶瓷茶壶套装，带红色花纹）",
    "count_visible": 3,
    "state": "状态描述（如：摆放整齐，未使用）"
  }},
  "intents": [
    {{ "intent_variants": {{ "formal_zh": "...", "formal_en": "...",
                             "natural_zh": "...", "natural_en": "...",
                             "casual_zh": "...", "casual_en": "...",
                             "emotional_zh": "...", "emotional_en": "..." }},
       "difficulty": "easy|medium|hard",
       "reasoning": "为什么这个意图指向 {target_category}（简短中文）" }},
    {{ "intent_variants": {{ "...": "..." }}, "difficulty": "...", "reasoning": "..." }},
    {{ "intent_variants": {{ "...": "..." }}, "difficulty": "...", "reasoning": "..." }},
    {{ "intent_variants": {{ "...": "..." }}, "difficulty": "...", "reasoning": "..." }}
  ]
}}

**重要**：输出必须是**一个**完整 JSON 对象，intents 数组长度**恰好 4**。不要输出解释文字、markdown 代码块标记、JSON 注释或多个 JSON 对象。

## 示例（target_category=wine_set, surface=tablecloth, room=餐厅）

```json
{{
  "specific_description": {{
    "object": "4 套白色陶瓷酒具，配红木托盘",
    "count_visible": 4,
    "state": "整齐摆放，似乎刚布置好"
  }},
  "intents": [
    {{
      "intent_variants": {{
        "formal_zh": "朋友马上要来做客了，我想好好准备一下",
        "formal_en": "Friends are coming over soon, I'd like to prepare things properly",
        "natural_zh": "要接待今晚的访客",
        "natural_en": "Getting ready for tonight's guests",
        "casual_zh": "客人要来",
        "casual_en": "Guests coming",
        "emotional_zh": "今晚想搞点仪式感，让聚会更特别",
        "emotional_en": "Want to add a festive touch tonight to make the gathering special"
      }},
      "difficulty": "medium",
      "reasoning": "接待客人的场景需要饮品器具"
    }}
  ]
}}
```

## 反例（会被过滤）
- ❌ "想拿个酒杯" — 含 "酒"
- ❌ "口渴想喝点东西" — 含 "喝"
- ❌ "读本书" — 含 "读"+"书"
- ❌ "清理一下房间" — 太笼统
"""


def build_objects_list(objects_on_surface):
    """Format objects list for prompt, including on/in relation."""
    lines = []
    for obj in objects_on_surface:
        oid = obj["object_id"]
        cat = obj["category"]
        rel = obj.get("relation", "on")
        rel_hint = "ON top" if rel == "on" else ("INSIDE" if rel == "in" else "ABOVE/MOUNTED")
        lines.append(f"  - {oid} ({cat}) [relation: {rel} → {rel_hint}]")
    return "\n".join(lines) if lines else "  (none)"


def build_room_context(room_dict, room_name, object_dict, exclude_ids=None):
    """List room's notable categories (excluding structural elements)."""
    if not room_dict or room_name not in room_dict:
        return "  (unknown)"

    exclude = set(exclude_ids or [])
    structural = {"wall", "floor", "ceiling", "door", "window", "curtain", "doorsill", "entrance"}

    items = []
    seen = set()
    for oid in room_dict[room_name]:
        if oid in exclude:
            continue
        cat = oid.split("_")[0] if "/" in oid else oid.split("_")[0]
        if cat in structural or cat in seen:
            continue
        seen.add(cat)
        items.append(cat)
    items = items[:15]
    return "  " + ", ".join(items) if items else "  (none)"


def build_spatial_relationships(objects_on_surface, object_dict):
    """Build spatial relationship descriptions."""
    lines = []
    for obj in objects_on_surface:
        oid = obj["object_id"]
        obj_data = object_dict.get(oid, {})
        nearby = obj_data.get("nearby_objects", {})
        cat = obj["category"]
        for nid, rel_info in nearby.items():
            rel = rel_info[0] if isinstance(rel_info, list) else rel_info
            ncat = nid.split("_")[0] if "/" in nid else nid.split("_")[0]
            lines.append(f"  {cat} is {rel} {ncat}")
    return "\n".join(lines[:10]) if lines else "  (none)"


def format_per_category_prompt(manifest_entry, surface_targets, room_dict, object_dict):
    """Format prompt for a per-category manifest entry.

    manifest_entry: dict from capture_manifest.json with target_category,
                    target_representative, all_targets_in_category fields
    surface_targets: matching entry from surface_targets.json (full object info)
    """
    surface_cat = manifest_entry["surface_category"]
    target_category = manifest_entry["target_category"]
    target_count = len(manifest_entry["all_targets_in_category"])
    room_name = manifest_entry["room"]

    # Get all categories on this surface
    all_cats = []
    seen = set()
    for o in surface_targets["objects_on_surface"]:
        c = o["category"]
        if c not in seen:
            seen.add(c)
            all_cats.append(c)

    # Find relation (on/above) for the target category
    target_rel = "on"
    for o in surface_targets["objects_on_surface"]:
        if o["category"] == target_category:
            target_rel = o.get("relation", "on")
            break

    room_context = build_room_context(
        room_dict, room_name, object_dict,
        exclude_ids=[o["object_id"] for o in surface_targets["objects_on_surface"]],
    )
    spatial = build_spatial_relationships(surface_targets["objects_on_surface"], object_dict)

    return INTENT_GENERATION_PROMPT.format(
        room_type=room_name,
        surface_category=surface_cat,
        target_category=target_category,
        target_relation=target_rel,
        target_count=target_count,
        all_categories_on_surface=", ".join(all_cats),
        room_context=room_context,
        spatial_relationships=spatial,
    )


# Backward-compat: old per-surface format
def format_prompt(surface, room_dict, object_dict):
    """Legacy: per-surface prompt (multiple targets per call). Deprecated."""
    objects_list = build_objects_list(surface["objects_on_surface"])
    obj_cats = []
    seen = set()
    for o in surface["objects_on_surface"]:
        c = o["category"]
        if c not in seen:
            seen.add(c)
            obj_cats.append(c)
    return INTENT_GENERATION_PROMPT.format(
        room_type=surface["room"],
        surface_category=surface["surface_category"],
        target_category=", ".join(obj_cats),
        target_relation="mixed",
        target_count=len(surface["objects_on_surface"]),
        all_categories_on_surface=", ".join(obj_cats),
        room_context="(see scene graph)",
        spatial_relationships="(see scene graph)",
    )
