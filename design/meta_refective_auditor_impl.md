下面给你一版**偏工程落地**、可以融进现有 video agent 的 **Meta-Reflective Auditor（证自证分）prototype 方案**。目标是：**尽量少改主干**，把它接成一个“围绕已有 agent 的外环”，先跑通，再迭代。

---

# 1. 设计目标（工程版）

你现有大概率已经有这些中的一部分：

- 视频感知：抽帧 / video encoder / object detector / tracker / OCR
- 主 agent：接收视频 + query，输出 answer / plan / action
- 或者已经有简单 reflection：对失败或低置信情况做自省

我们现在要加的，不是重做一个大系统，而是加一个：

> **二阶审计器（Meta-Reflective Auditor, MRA）**  
> 专门检查 agent 的 reflection 是否值得信。

工程目标：

1. **不重写主 agent**
2. **不依赖复杂新训练**
3. **先支持少数固定 error types 和 interventions**
4. **优先提升：视频感知错误定位、无根据反思过滤、局部重看策略**
5. **最后输出可用于你们 agent 决策的 meta-trust / next-step**

---

# 2. 原型系统最小闭环

## 2.1 你需要的最小组件

我建议 MVP 只保留这 6 个：

### C1. Base Video Agent
现有 agent 即可。输入视频和 query，输出：
- answer
- claim(s)
- confidence
- 支持时间段（如果能给就更好）

### C2. Reflector
一个轻量自省器，最好就是你现有 agent 或另一个 prompt 调同一模型。
输出固定 JSON：
- 哪个 claim 可疑
- 错因类型
- 关键时间段
- 修复建议
- 预期变化

### C3. Evidence Extractor
从已有 perception pipeline 拿证据：
- sampled frames
- object tracks
- detector scores
- OCR results
- temporal span features
- uncertainty proxy

### C4. Meta-Reflective Auditor
核心模块。基于 reflection + evidence 做：
- 可信度评分
- 竞争假设生成
- 干预选择

### C5. Intervention Executor
先只支持少数操作：
- dense resample local frames
- zoom local region
- rerun detector / tracker / OCR
- evidence-only rereason

### C6. Delta Evaluator
比较 intervention 前后的：
- answer
- claim confidence
- visual support
- reflection consistency

---

## 2.2 最小闭环流程

```text
video, query
   ↓
Base Agent
   ↓
answer + claims + trace
   ↓
Reflector
   ↓
reflection(JSON)
   ↓
Meta-Reflective Auditor
   ↓
if trust high:
    accept / revise / abstain
else:
    choose intervention
       ↓
    Intervention Executor
       ↓
    rerun local perception + reasoning
       ↓
    Delta Evaluator
       ↓
    update meta-trust
       ↓
final decision
```

---

# 3. 建议你先支持的任务类型

为了让 prototype 容易落地，先选这类视频任务：

## 优先级最高：视频 QA / 事件判断
例如：
- 这个人有没有拿起杯子？
- 门是否被打开？
- 物体是否发生接触？
- 动作发生在前还是后？

因为这类任务：
- claim 相对明确
- 可绑定时间段
- intervention 效果明显
- 不需要一开始就接复杂 action policy

## 第二阶段：视频规划 / embodied action
例如：
- 接下来该抓哪个物体
- 当前状态是否允许执行动作
- 是否需要换视角

---

# 4. MVP 的错误类型 taxonomy

先严格限制，不要一开始太多。

我建议第一版只做 5 类：

```python
ERROR_TYPES = [
    "visual_ambiguity",        # 看不清/遮挡/模糊
    "temporal_boundary_error", # 事件起止判断错
    "tracking_failure",        # 对象跟踪断裂/错跟
    "ocr_ambiguity",           # 文本区域不清
    "language_prior_bias"      # 语言先验脑补
]
```

解释：

### 4.1 visual_ambiguity
适用于：
- 手与物体接触看不清
- 遮挡
- 低分辨
- 运动模糊

### 4.2 temporal_boundary_error
适用于：
- 动作是否完成
- 是接近还是已经发生
- 前后顺序搞错

### 4.3 tracking_failure
适用于：
- 物体 track 丢失
- 错 object association
- 多个相似物体混淆

### 4.4 ocr_ambiguity
适用于视频里有文字、面板、数字、GUI。

### 4.5 language_prior_bias
适用于：
- 视觉支持弱，但回答用了高频常识模板
- 反思里把问题归因得很虚

---

# 5. MVP 的 intervention 集合

第一版我建议只支持 4 类：

```python
INTERVENTIONS = [
    "dense_frame_resample",
    "zoom_region",
    "rerun_tracker_or_detector",
    "evidence_only_rereason"
]
```

---

## 5.1 dense_frame_resample
对某时间段重新更密地抽帧。

适合：
- temporal boundary error
- 动作过快
- 原来采样太稀

---

## 5.2 zoom_region
对局部区域做高分辨 crop，再抽特征。

适合：
- visual ambiguity
- OCR ambiguity
- 小目标接触判断

---

## 5.3 rerun_tracker_or_detector
在局部 span 上重跑：
- detector
- tracker
- contact detector
- OCR

适合：
- tracking failure
- 接触判断类

---

## 5.4 evidence_only_rereason
重新让 LLM/agent 推理，但强制：
- 只能使用给定证据
- 不允许脑补没有 grounding 的内容
- 必须引用 frame span / object track / OCR span

适合：
- language prior bias
- 也适合作为便宜的第一道干预

---

# 6. 数据结构：一定要先定协议

你后续要融进现有 agent，所以我建议先把几个 JSON/字典结构定死。

---

## 6.1 Base agent 输出协议

```python
BaseOutput = {
    "answer": "yes",
    "answer_confidence": 0.74,
    "claims": [
        {
            "claim_id": "c1",
            "type": "event",
            "text": "person picks up the cup",
            "span": [42, 58],
            "objects": ["person#1", "cup#1"],
            "confidence": 0.74,
            "support_refs": {
                "frames": [42, 45, 48, 52],
                "tracks": ["hand#1", "cup#1"],
                "extra": ["event_candidate:e1"]
            }
        }
    ],
    "trace": {
        "sampled_frames": [36, 40, 44, 48, 52, 56],
        "used_modules": ["video_llm"],
        "notes": "coarse temporal understanding"
    }
}
```

如果你们现有 agent 没有 `claims`，就从 answer/rationale 里抽一个最核心 claim 先凑出来。

---

## 6.2 Reflection 输出协议

强制 JSON，不要自由文本。

```python
ReflectionOutput = {
    "overall_self_confidence": 0.61,
    "claim_reviews": [
        {
            "claim_id": "c1",
            "status": "possibly_wrong",
            "error_type": "visual_ambiguity",
            "time_span": [42, 48],
            "region_hint": "hand_cup_contact_region",
            "objects": ["hand#1", "cup#1"],
            "evidence_gap": "contact relation unclear",
            "proposed_fix": [
                "zoom_region",
                "rerun_tracker_or_detector"
            ],
            "expected_change": "if there is no stable grasp evidence, confidence for pickup should decrease"
        }
    ],
    "global_risk": "medium"
}
```

关键字段：
- `error_type`
- `time_span`
- `proposed_fix`
- `expected_change`

没有这几个，后面审计不起来。

---

## 6.3 Evidence Bundle 协议

这是 MRA 的输入之一。建议从 perception 模块统一导出：

```python
EvidenceBundle = {
    "frame_meta": {
        42: {"blur": 0.62, "occlusion": 0.31},
        43: {"blur": 0.59, "occlusion": 0.28}
    },
    "tracks": {
        "hand#1": {
            "span": [30, 60],
            "avg_conf": 0.78,
            "conf_by_frame": {42: 0.61, 43: 0.58, 44: 0.55}
        },
        "cup#1": {
            "span": [1, 100],
            "avg_conf": 0.84,
            "conf_by_frame": {42: 0.83, 43: 0.81, 44: 0.79}
        }
    },
    "event_candidates": [
        {
            "type": "pickup",
            "span": [42, 58],
            "confidence": 0.57
        }
    ],
    "ocr_spans": [],
    "support_trace": {
        "c1": {
            "frames": [42, 45, 48, 52],
            "tracks": ["hand#1", "cup#1"]
        }
    }
}
```

---

# 7. 模块拆分与代码目录建议

你后续要融进现有 agent，我建议单独包成一个模块目录。

```text
meta_reflective_auditor/
├── __init__.py
├── schemas.py              # 定义 BaseOutput, ReflectionOutput, EvidenceBundle 等
├── reflector.py            # 调模型生成 structured reflection
├── evidence_collector.py   # 从现有感知结果中组装 EvidenceBundle
├── parser.py               # reflection -> internal claim graph
├── auditor.py              # 核心 MRA 逻辑
├── hypothesis.py           # competing hypotheses 生成与评分
├── intervention.py         # intervention 选择与执行接口
├── delta_eval.py           # 干预前后对比
├── decision.py             # 最终 accept / revise / abstain 决策
└── utils.py
```

如果你们工程已经有 agent pipeline，可以把它作为一个外部 service 或 pipeline stage 插进去。

---

# 8. 运行时核心逻辑：可执行版

下面给你一版更像 Python 代码的流程。

---

## 8.1 主入口

```python
def run_with_meta_reflection(video, query, base_agent, tools, config):
    # 1. base inference
    base_out = base_agent.infer(video, query)

    # 2. collect evidence
    evidence = collect_evidence(video, base_out, tools, config)

    # 3. structured reflection
    reflection = generate_reflection(video, query, base_out, evidence, config)

    # 4. meta audit
    audit = run_meta_audit(video, query, base_out, reflection, evidence, config)

    # 5. direct decision or intervention
    if should_accept_without_intervention(audit, base_out, config):
        return finalize_result(base_out, reflection, audit, status="accepted_without_intervention")

    # 6. choose and execute intervention
    intervention = select_best_intervention(audit, config)
    if intervention is None:
        return finalize_result(base_out, reflection, audit, status="no_intervention_possible")

    updated_bundle = execute_intervention(video, query, base_out, evidence, intervention, tools, config)

    # 7. localized rerun
    new_out = rerun_local_reasoning(video, query, base_out, updated_bundle, intervention, base_agent, config)

    # 8. reflect again
    new_reflection = generate_reflection(video, query, new_out, updated_bundle, config)

    # 9. compare
    delta = evaluate_delta(base_out, reflection, new_out, new_reflection, updated_bundle, intervention)

    # 10. final meta audit with delta
    final_audit = update_audit_with_delta(audit, delta, new_reflection, config)

    # 11. final decision
    final = finalize_decision(base_out, new_out, reflection, new_reflection, final_audit, config)
    return final
```

---

# 9. 每个函数怎么实现

---

## 9.1 `collect_evidence`

### 输入
- 原视频
- base_out
- tools（detector/tracker/OCR 等）

### 输出
- `EvidenceBundle`

### 最低实现
如果你们已有 perception 中间结果，直接整理。  
如果没有，就补这几个 proxy：

- 抽 base_out claim 对应 span 的帧
- 跑 detector/tracker
- 记录置信度
- 记录局部 blur / motion proxy

```python
def collect_evidence(video, base_out, tools, config):
    spans = [c["span"] for c in base_out["claims"] if "span" in c]
    focus_span = merge_spans(spans)

    frames = sample_frames(video, focus_span, stride=config.frame_stride)
    frame_meta = estimate_frame_quality(frames)

    tracks = {}
    if tools.tracker is not None:
        tracks = tools.tracker.run(frames)

    event_candidates = []
    if tools.event_detector is not None:
        event_candidates = tools.event_detector.run(frames)

    return {
        "frame_meta": frame_meta,
        "tracks": tracks,
        "event_candidates": event_candidates,
        "support_trace": build_support_trace(base_out)
    }
```

---

## 9.2 `generate_reflection`

### 最低实现方式
直接 prompt 现有 LLM / MLLM，要求输出严格 JSON。

### Prompt 关键约束
- 只能从固定 error taxonomy 选
- 必须给 time_span
- 必须给 proposed_fix
- 必须给 expected_change
- 不知道就输出 `unknown`

```python
def generate_reflection(video, query, base_out, evidence, config):
    prompt = build_reflection_prompt(query, base_out, evidence, config)
    raw = call_llm(prompt)
    reflection = parse_json_safely(raw)
    reflection = normalize_reflection(reflection)
    return reflection
```

---

## 9.3 `run_meta_audit`

这是核心。第一版可不用太复杂，先做 rule-based + heuristic scoring。

### 核心输出
- groundedness
- attribution_validity
- fix_validity
- recommended_intervention
- meta_trust

```python
def run_meta_audit(video, query, base_out, reflection, evidence, config):
    groundedness = score_groundedness(reflection, evidence)
    attribution_validity = score_attribution_validity(reflection, base_out, evidence)
    fix_validity = score_fix_validity(reflection, evidence)

    hypotheses = build_simple_hypotheses(reflection, base_out, evidence)

    meta_trust = aggregate_scores(
        groundedness=groundedness,
        attribution_validity=attribution_validity,
        fix_validity=fix_validity
    )

    recommended = recommend_intervention(reflection, hypotheses, evidence, config)

    return {
        "groundedness": groundedness,
        "attribution_validity": attribution_validity,
        "fix_validity": fix_validity,
        "meta_trust": meta_trust,
        "hypotheses": hypotheses,
        "recommended_intervention": recommended
    }
```

---

# 10. 先做 rule-based scoring，最现实

你现在不是要研究最优理论，而是可执行原型。  
所以第一版我建议**规则 + heuristics**，不先搞复杂学习。

---

## 10.1 `score_groundedness`

判断 reflection 是否真的对上证据。

### 规则示例

#### 若 `error_type = visual_ambiguity`
检查：
- 指定 time_span 的 blur 是否高
- 是否存在 occlusion
- 对象 track conf 是否低
- claim 支持是否主要集中在该 span

#### 若 `error_type = temporal_boundary_error`
检查：
- base_out claim span 边界附近 event confidence 是否低
- 原采样是否过稀

#### 若 `error_type = tracking_failure`
检查：
- 指定对象 track conf 是否下降/断裂

#### 若 `error_type = language_prior_bias`
检查：
- 原 claim support refs 是否薄弱
- evidence-only prompt 下是否容易变化（可后做）

```python
def score_groundedness(reflection, evidence):
    scores = []
    for review in reflection["claim_reviews"]:
        et = review["error_type"]
        span = review["time_span"]

        if et == "visual_ambiguity":
            blur = avg_blur(evidence["frame_meta"], span)
            occ  = avg_occlusion(evidence["frame_meta"], span)
            trk  = avg_track_drop(evidence["tracks"], review.get("objects", []), span)
            s = 0.4*blur + 0.3*occ + 0.3*trk

        elif et == "temporal_boundary_error":
            s = temporal_boundary_uncertainty(evidence, span)

        elif et == "tracking_failure":
            s = avg_track_drop(evidence["tracks"], review.get("objects", []), span)

        elif et == "ocr_ambiguity":
            s = ocr_uncertainty(evidence, span)

        elif et == "language_prior_bias":
            s = weak_support_score(evidence, review["claim_id"])

        else:
            s = 0.2

        scores.append(clamp(s, 0, 1))

    return sum(scores)/max(len(scores), 1)
```

---

## 10.2 `score_attribution_validity`

检查“这个错因是不是和 claim 类型匹配”。

### 规则示例
如果 claim 是 `pickup event`：
- `visual_ambiguity`、`temporal_boundary_error`、`tracking_failure` 一般更合理
- `ocr_ambiguity` 几乎无关 → 低分

```python
def score_attribution_validity(reflection, base_out, evidence):
    claim_map = {c["claim_id"]: c for c in base_out["claims"]}
    vals = []

    for review in reflection["claim_reviews"]:
        claim = claim_map.get(review["claim_id"], {})
        ctype = claim.get("type", "unknown")
        et = review["error_type"]

        s = compatibility_score(ctype, et)

        # 如果有 objects 且在 claim objects 里，略加分
        if overlap(review.get("objects", []), claim.get("objects", [])):
            s += 0.1

        # 如果 time_span 与 claim span 有重叠，略加分
        if span_overlap(review.get("time_span"), claim.get("span")) > 0:
            s += 0.1

        vals.append(clamp(s, 0, 1))

    return sum(vals)/max(len(vals), 1)
```

---

## 10.3 `score_fix_validity`

检查 proposed_fix 与 error_type 是否匹配。

### 例如
- visual_ambiguity -> zoom_region / rerun_detector 高分
- temporal_boundary_error -> dense_frame_resample 高分
- language_prior_bias -> evidence_only_rereason 高分

```python
def score_fix_validity(reflection, evidence):
    vals = []
    for review in reflection["claim_reviews"]:
        et = review["error_type"]
        fixes = review.get("proposed_fix", [])
        vals.append(avg([fix_match_score(et, f) for f in fixes]) if fixes else 0.2)
    return sum(vals)/max(len(vals), 1)
```

---

# 11. intervention 选择逻辑：直接能跑的版本

第一版别做复杂优化，直接规则优先级即可。

```python
def recommend_intervention(reflection, hypotheses, evidence, config):
    candidates = []

    for review in reflection["claim_reviews"]:
        et = review["error_type"]
        fixes = review.get("proposed_fix", [])

        for f in fixes:
            score = 0.0

            if f == "zoom_region" and et == "visual_ambiguity":
                score = 0.9
            elif f == "dense_frame_resample" and et == "temporal_boundary_error":
                score = 0.9
            elif f == "rerun_tracker_or_detector" and et in ["tracking_failure", "visual_ambiguity"]:
                score = 0.85
            elif f == "evidence_only_rereason" and et == "language_prior_bias":
                score = 0.8
            else:
                score = 0.4

            candidates.append({
                "type": f,
                "score": score,
                "review": review
            })

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates[0]
```

---

# 12. execute_intervention：最小可行实现

---

## 12.1 dense_frame_resample

```python
def do_dense_frame_resample(video, span, config):
    return sample_frames(video, span, stride=1)
```

---

## 12.2 zoom_region

如果还没有 region detector，第一版可以粗糙些：
- 基于 objects 的 bbox 并集
- 或基于手工 region_hint

```python
def do_zoom_region(video, span, tracks, objects, config):
    region = estimate_region_from_tracks(tracks, objects, span)
    zoomed_frames = crop_and_resize(video, span, region, size=config.zoom_size)
    return {
        "zoomed_frames": zoomed_frames,
        "region": region
    }
```

---

## 12.3 rerun_tracker_or_detector

```python
def do_rerun_tracker_or_detector(video, span, tools, config):
    frames = sample_frames(video, span, stride=1)
    tracks = tools.tracker.run(frames) if tools.tracker else {}
    dets = tools.detector.run(frames) if tools.detector else {}
    return {
        "frames": frames,
        "tracks": tracks,
        "detections": dets
    }
```

---

## 12.4 evidence_only_rereason

这个很有性价比。  
就是重新 prompt LLM，只给结构化证据，不给长自由文本。

```python
def do_evidence_only_rereason(query, claim, evidence, llm):
    prompt = build_evidence_only_prompt(query, claim, evidence)
    return llm(prompt)
```

规则：
- 只能引用提供证据
- 不能添加证据之外的事实
- 若证据不足，必须输出 uncertain

---

# 13. rerun_local_reasoning：不要全局重跑

这一点非常重要。  
否则干预引入新的全局漂移，难判断是否是 reflection 被证实。

---

## 13.1 只更新被 targeting 的 claim

```python
def rerun_local_reasoning(video, query, base_out, updated_bundle, intervention, base_agent, config):
    target_claim = intervention["review"]["claim_id"]

    # 只针对 target claim 构造局部问题
    local_query = build_local_query(query, base_out, target_claim, updated_bundle)

    # 可调用同一个 base_agent，但传入局部证据
    local_answer = base_agent.local_infer(local_query, updated_bundle)

    # 合并回原输出
    new_out = copy.deepcopy(base_out)
    update_claim(new_out, target_claim, local_answer)
    recompute_overall_answer(new_out)

    return new_out
```

---

# 14. Delta Evaluator：最关键的“证”来自这里

你真正想验证的是：

> 原 reflection 说的修复，执行后是不是按它预期那样变化？

所以 delta 不 فقط 看“对没对”，而要看**是否符合 expected_change**。

---

## 14.1 delta 结构

```python
Delta = {
    "answer_changed": True,
    "claim_conf_delta": -0.33,
    "support_strength_delta": +0.21,
    "expected_change_matched": True,
    "reflection_became_more_specific": True
}
```

---

## 14.2 计算逻辑

```python
def evaluate_delta(base_out, reflection, new_out, new_reflection, updated_bundle, intervention):
    target_claim_id = intervention["review"]["claim_id"]

    old_claim = get_claim(base_out, target_claim_id)
    new_claim = get_claim(new_out, target_claim_id)

    conf_delta = new_claim["confidence"] - old_claim["confidence"]

    expected = intervention["review"].get("expected_change", "").lower()

    matched = match_expected_change(expected, old_claim, new_claim, updated_bundle)

    return {
        "answer_changed": base_out["answer"] != new_out["answer"],
        "claim_conf_delta": conf_delta,
        "expected_change_matched": matched,
        "reflection_became_more_specific": reflection_specificity(new_reflection) >= reflection_specificity(reflection)
    }
```

---

# 15. final decision：一版简单实用的规则

```python
def finalize_decision(base_out, new_out, reflection, new_reflection, final_audit, config):
    mts = final_audit["meta_trust"]

    if mts >= 0.75 and new_out["answer"] != base_out["answer"]:
        return {
            "status": "accept_revised_answer",
            "final_output": new_out,
            "final_reflection": new_reflection,
            "audit": final_audit
        }

    if mts >= 0.75 and new_out["answer"] == base_out["answer"]:
        return {
            "status": "accept_original_with_higher_confidence",
            "final_output": new_out,
            "final_reflection": new_reflection,
            "audit": final_audit
        }

    if 0.45 <= mts < 0.75:
        return {
            "status": "keep_uncertain",
            "final_output": mark_uncertain(new_out),
            "final_reflection": new_reflection,
            "audit": final_audit
        }

    return {
        "status": "abstain_or_escalate",
        "final_output": abstain_output(base_out),
        "final_reflection": new_reflection,
        "audit": final_audit
    }
```

---

# 16. 推荐配置：第一版直接能上手的参数

```python
config = {
    "frame_stride": 4,
    "max_intervention_rounds": 1,   # 第一版只做 1 次干预
    "zoom_size": 336,
    "meta_trust_accept": 0.75,
    "meta_trust_uncertain": 0.45,
    "min_claim_conf_for_reflect": 0.80,  # 低于这个值才触发 reflection
    "supported_error_types": [
        "visual_ambiguity",
        "temporal_boundary_error",
        "tracking_failure",
        "ocr_ambiguity",
        "language_prior_bias"
    ]
}
```

建议第一版：
- **只允许 1 轮 intervention**
- 只针对 **top-1 target claim**
- 只支持 **一个 selected intervention**

这样最容易集成。

---

# 17. 接到现有 agent 上的最小改动点

如果你们已有 video agent，不想大改，我建议只加这 4 个 hook。

---

## Hook 1: Base output standardization
把现有输出包成：

- `answer`
- `answer_confidence`
- `claims`
- `trace`

哪怕 `claims` 只有一个。

---

## Hook 2: Reflection call
在这两种情况下触发：
- 低置信
- 高风险任务
- 或输出不满足某验证器

```python
if base_out["answer_confidence"] < threshold or verification_failed:
    reflection = generate_reflection(...)
```

---

## Hook 3: Perception evidence API
从现有感知模块暴露：
- frame quality
- tracks
- OCR
- event candidates

即使没有全部，也至少给：
- 关键帧
- detector/tracker 置信度
- 原 answer 对应时间段

---

## Hook 4: Local rerun API
提供一个局部重推理接口：

```python
base_agent.local_infer(local_query, local_evidence)
```

如果没有，就用全局 infer + 局部 prompt 近似。

---

# 18. 第一阶段最值得监控的日志

你后续一定会需要 debug，所以我建议从第一天就把下面这些打日志。

---

## 每次运行记录：
- base answer / confidence
- target claim
- reflection error_type
- reflection span
- groundedness score
- attribution_validity
- selected intervention
- intervention result
- expected_change 是否匹配
- final status

### 示例日志
```python
{
  "qid": "videoqa_00123",
  "base_answer": "yes",
  "base_conf": 0.74,
  "target_claim": "person picks up cup",
  "reflection_error_type": "visual_ambiguity",
  "reflection_span": [42, 48],
  "groundedness": 0.79,
  "attribution_validity": 0.82,
  "selected_intervention": "zoom_region",
  "delta_conf": -0.31,
  "expected_change_matched": True,
  "final_status": "accept_revised_answer"
}
```

这个日志会非常有用，你之后可以直接拿来分析：
- 哪些 reflection 常是假
- 哪些 intervention 有用
- 哪些 error type 最常出现

---

# 19. 第一版不要做什么

为了可执行，我建议你一开始不要上这些：

1. 不要多轮复杂 debate
2. 不要太多 error types
3. 不要全局 MCTS 式搜索
4. 不要复杂 world model 更新
5. 不要先做训练版 meta-auditor
6. 不要追求所有任务统一

先把：
- 结构化 reflection
- groundedness 审计
- 一个 intervention
- delta 验证
跑通，就已经很值钱了。

---

# 20. 一个非常实用的落地顺序

这是我最建议你的实施顺序。

---

## Phase 1：最小闭环
实现：
- BaseOutput 协议
- Reflection JSON
- Rule-based audit
- 1 次 intervention
- final decision

目标：
- 能在视频 QA 上跑通

---

## Phase 2：加 perception hooks
补：
- track conf
- blur/occlusion
- event candidates
- OCR uncertainty

目标：
- groundedness 更靠谱

---

## Phase 3：加 evidence-only rereason
这是低成本高收益项。  
很多语言脑补会立刻暴露。

---

## Phase 4：加历史记忆
记录：
- 哪类 reflection 值得信
- 哪类 intervention 有效

目标：
- 更稳定的 meta-trust

---

# 21. 给你一版最小 prompt 模板

## 21.1 Reflection prompt

你可以直接让一个 LLM 输出：

```text
You are a self-assessment module for a video agent.

Given:
1. the user query
2. the agent's answer
3. the agent's claims
4. available evidence summary

Your job is to review whether the answer may be wrong.

You must output JSON only.

Rules:
- Choose error_type only from:
  ["visual_ambiguity", "temporal_boundary_error", "tracking_failure", "ocr_ambiguity", "language_prior_bias"]
- Each suspicious claim must include:
  claim_id, error_type, time_span, objects, evidence_gap, proposed_fix, expected_change
- proposed_fix must be chosen from:
  ["dense_frame_resample", "zoom_region", "rerun_tracker_or_detector", "evidence_only_rereason"]
- If unsure, still output the most likely error_type and mark status as "uncertain_review"
- Do not write explanation outside JSON
```

---

## 21.2 Evidence-only rereason prompt

```text
You are a strictly grounded reasoning module.

You are given:
- a question
- one target claim
- only the following verified evidence

Rules:
- Use only the provided evidence
- Do not infer facts not directly supported
- If evidence is insufficient, answer "uncertain"
- Return JSON with:
  answer, confidence, evidence_used, unsupported_parts
```

---

# 22. 我建议的“第一版判成败标准”

不是最终 accuracy，而是这几个：

1. reflection 结构化输出稳定率
2. 有多少反思能被 groundedness 审计有效筛掉
3. intervention 后，expected_change 匹配率
4. 对明显 hallucination case 的修正率
5. 最终是否减少“高置信错误”