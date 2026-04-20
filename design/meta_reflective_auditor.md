

我先给一句核心定义：

> **证自证分模块**：不是判断 agent 的答案对不对，而是判断 agent 对自己当前认知、错误归因、修正建议的“自我说明”是否可信，并据此决定是否接受该反思、触发额外感知/工具/重规划，或进入更强验证流程。

---

---
# 0. 适用场景与目标
---

## 0.1 场景
面向 **video agent**，例如：

- 视频问答（Video QA）
- 视频理解与事件推理
- 视频导航 / embodied agent
- 操作型 agent（根据视频观测执行动作）
- GUI/video control agent
- surveillance/egocentric video 分析 agent

agent 主流程通常是：

- **Perception**：看视频，得到对象/事件/状态
- **Plan**：形成回答、动作计划、事件解释
- **Action**：执行动作或输出答案
- **Reflection**：反思自己是否看错、想错、做错

你要加的是：

- **Meta-Reflection Auditor / 证自证分模块**

---

## 0.2 它解决什么问题

现有 video agent 常见问题：

1. **看得不全**：关键帧没看清、时间段没覆盖
2. **想得太快**：语言先验替代真实时序理解
3. **反思像在编理由**：事后找个看似合理的错因
4. **知道自己不确定，但不知道这种不确定是否可信**
5. **修正策略无效**：说该看更多帧，但看了也没改善

所以证自证分模块要回答三个核心问题：

### Q1. 这份 reflection 靠不靠谱？
比如 agent 说：
- “我错在没看到第 42-48 帧中的手部接触”
这句话可信么？

### Q2. 这份 reflection 值不值得执行其修正建议？
比如它建议：
- 放大某时间段
- 重新做 tracking
- 调用 OCR
- 改走另一条计划
这些建议值得花算力执行吗？

### Q3. 即使不直接知道真值，能否判断当前自省是否“像是真的抓到了错因”？
也就是一种**二阶认识可靠性估计**。

---

---
# 1. 整体系统结构
---

我先给一个模块图式的口头结构：

## 主 agent 流程
1. Video observation \(V\)
2. Perception 模块输出世界表征 \(S\)
3. Planner/Reasoner 生成候选答案/动作/计划 \(Y\)
4. Reflection 模块生成自评 \(R\)

## 新增：证自证分模块 \(MRA = Meta-Reflective Auditor\)
输入：
- 视频与中间表征
- 初始输出 \(Y\)
- reflection \(R\)
- 工具调用记录 / 证据引用 / 时间片段
- 历史记忆（类似过去相似错误与修正成效）

输出：
- 自省可信度 `trust_reflection`
- 错因归因可信度 `trust_error_attribution`
- 修正建议可信度 `trust_fix`
- 推荐的下一步 `next_step`

候选 `next_step`：
- accept reflection and revise
- re-perceive selected frames
- run specialized tool
- ask for more context
- re-plan
- abstain / defer
- escalate to stronger verifier

---

---
# 2. 核心输入输出定义
---

为了可执行，我们先把数据结构定清楚。

## 2.1 视频输入

```text
V = {f_1, f_2, ..., f_T}
```

可附带：
- 音频流
- 时间戳
- action history
- environment state
- text query / user instruction

---

## 2.2 perception 输出的基础表征 S

建议不是单个 embedding，而是结构化的：

```text
S = {
  frame_tokens,
  temporal_segments,
  object_tracks,
  scene_graphs_over_time,
  event_candidates,
  OCR_spans,
  uncertainty_map,
  saliency_trace,
  memory_retrievals
}
```

最低配也应该有：

- 关键帧列表
- temporal segments
- object tracks
- event hypotheses
- 每个 hypotheses 的置信度
- 对应证据时间段

---

## 2.3 主输出 Y

按任务不同可能是：
- QA answer
- action sequence
- plan
- event label
- state estimate

为了统一，可以写成 claim 集合：

```text
Y = {c_1, c_2, ..., c_n}
```

每个 claim 包含：
- 内容
- 置信度
- 支持证据（如果有）
- 对应时间区间
- 依赖对象/事件

例如：

```text
c_i = {
  text: "person picks up the cup",
  time_span: [42, 58],
  support_tracks: [hand#1, cup#3],
  confidence: 0.74
}
```

---

## 2.4 reflection 输出 R

这个非常关键。reflection 不要只是自由文本，应该结构化：

```text
R = {
  self_confidence,
  uncertainty_sources,
  error_attribution,
  unsupported_claims,
  missing_evidence,
  proposed_interventions,
  revised_hypotheses,
  expected_gain
}
```

更细一点：

### A. self_confidence
- overall confidence
- per-claim confidence

### B. uncertainty_sources
从预定义集合中选：
- low_resolution
- occlusion
- temporal_gap
- tracking_failure
- OCR_ambiguity
- audio_conflict
- language_prior_dominance
- weak causal link
- insufficient context

### C. error_attribution
它认为错误主要在哪里：
- perception
- temporal integration
- object tracking
- relation reasoning
- planning
- memory retrieval
- action execution

### D. unsupported_claims
指出哪些原 claim 支持不足

### E. proposed_interventions
建议的修正动作，例如：
- inspect frames [a:b]
- zoom region r in frames [a:b]
- re-run tracking on object k
- request OCR for frames [a:b]
- retrieve earlier context
- delay answer
- ask clarifying question

### F. expected_gain
每个 intervention 预计能修复哪些 claim

---

## 2.5 证自证分模块输出 A

```text
A = {
  trust_overall,
  trust_per_reflection_field,
  attribution_validity,
  intervention_priority,
  intervention_budget,
  next_step,
  audit_report
}
```

例如：

```text
A = {
  trust_overall: 0.62,
  attribution_validity: {
    perception: 0.81,
    temporal_integration: 0.45
  },
  intervention_priority: [
    zoom(frames 42-48, region hand-cup),
    re-run contact detector,
    defer global replan
  ],
  next_step: "targeted_reperception"
}
```

---

---
# 3. 证自证分模块的核心思想
---

它不直接判断“答案对不对”，而是判断 reflection 的四类性质：

## 3.1 证据落地性 Groundedness
reflection 说的那些错因、证据缺口、关键时间片，是否真的和可观察数据对应？

例如：
- 它说“42-48 帧有遮挡”
模块去看：
  - 那几帧 uncertainty 是否高？
  - 目标区域检测质量是否低？
  - track 是否断裂？
  - resolution 是否不足？

---

## 3.2 行为预测性 Predictive Validity
reflection 提出的解释是否能预测：
- 哪些 intervention 会有效
- 修改后是否有机会改善结果

例如：
- 它说“错因是 OCR 模糊”
那么调用 OCR 超分辨或高分辨局部后，应该更可能改善

---

## 3.3 跨视角一致性 Cross-Check Consistency
reflection 是否与以下一致：
- perception trace
- planner trace
- action outcome
- 工具日志
- 历史相似案例

例如：
- 它说“我没看到杯子”
但 object tracker 明明持续追踪到了杯子，那这段 reflection 可疑

---

## 3.4 非模板化 Causal Specificity
reflection 是否真的指出具体因果点，而不是套话。

比如：
- “我可能缺乏信息” → 太泛
- “在 42-48 帧右下角手与杯子接触区域分辨率不足，导致把‘靠近’误判成‘抓取’” → 高质量

---

---
# 4. 模块内部组成
---

建议把证自证分模块拆成五个子器件。

---

## 4.1 Claim-Reflection Parser
把自由文本或半结构化 reflection 转成 machine-checkable 的 claim 图。

输入：
- reflection 文本/结构化字段
- 原输出 claims

输出：
- `reflection_claim_graph`

例如把：

> “我可能误把手靠近杯子看成拿起，因为 42-48 帧接触证据不清楚，建议放大该区域再检查。”

解析成：

```text
RCG = {
  attribution_claims: [
    {type: "temporal_perception_error", target_claim: c_2, span:[42,48]}
  ],
  evidence_claims: [
    {type: "contact_evidence_missing", objects:[hand#1,cup#3], span:[42,48]}
  ],
  intervention_claims: [
    {type: "zoom_and_recheck", span:[42,48], region:"hand-cup"}
  ],
  causal_links: [
    perception_blur -> contact_ambiguity -> wrong_pickup_claim
  ]
}
```

作用：
让 reflection 变得可验证。

---

## 4.2 Evidence Auditor
检查 reflection 是否 grounded。

输入：
- 视频
- 感知中间结果 S
- reflection claim graph

输出：
- 每条 reflection claim 的证据支持度

检查项包括：
1. 时间段是否存在
2. 该时间段是否真高不确定
3. 所指对象是否被检测/跟踪到
4. 所说区域是否真有分辨率问题/遮挡
5. 所说的“缺少证据”是否真的缺

输出示意：

```text
evidence_audit = {
  claim_r1: {support: 0.88, reason: "track break + low contact confidence"},
  claim_r2: {support: 0.31, reason: "OCR quality normal"},
}
```

---

## 4.3 Causal Plausibility Auditor
检查 reflection 的错因链条是否合理。

例如：
- “因为漏看一帧，所以整个 20 秒动作判断错了”
未必合理
- “因为接触关系不明确，所以‘靠近’与‘拿起’混淆”
更合理

它不要求绝对真，但要看是否符合任务结构和世界常识。

输入：
- task type
- claim graph
- world state graph
- action log / event graph

输出：
- 错因链条 plausibility
- proposed fix 是否和错因相匹配

例如：
- 如果错因是 tracking failure，建议却是“增加语言推理”，则匹配度低

---

## 4.4 Intervention Simulator / Selector
这是核心中的核心。

reflection 提出修复建议后，模块要估计：

- 哪个 intervention 最有信息增益？
- 哪个最可能证实/证伪该 reflection？
- 预算有限时先做哪个？

候选 intervention：
- 重看关键时间段
- 提高局部帧分辨率
- 重做 tracking
- 跑接触检测器
- 重算事件边界
- 拉长上下文窗口
- 检索前序记忆
- 请求外部工具
- 重新规划

它的选择原则是：

> 优先执行那些能最大程度验证 reflection 真伪、且成本合理的 intervention。

这非常像“证”：
不是只听你说，而是做一个最有鉴别力的动作来验证。

---

## 4.5 Reflection Reputation Memory
维护“某类反思 historically 是否靠谱”的长期记忆。

例如记录模式：
- “归因给 temporal gap” 在类似任务中成功率 63%
- “归因给 OCR 模糊” 在该视频分布中成功率 78%
- “归因给常识不足” 常是虚假反思，成功率仅 18%

这个 memory 不做最终裁决，但会调整 `trust_overall`。

---

---
# 5. 推理时主流程
---

下面给一个完整可执行流程。

---

## Step 1: 主 agent 输出初始结果
输入视频和任务：

```text
(V, q) -> S -> Y
```

其中：
- `S` 是感知与中间状态
- `Y` 是初始答案/计划/动作

---

## Step 2: Reflection 生成自省
reflection 模块输出：

```text
R = Reflect(V, S, Y, trace)
```

内容包括：
- 哪些 claim 可能错
- 为什么错
- 错在哪里
- 应如何修复
- 修复预计改善什么

---

## Step 3: 证自证分模块解析 reflection
```text
RCG = ParseReflection(R, Y)
```

得到结构化 claim graph。

---

## Step 4: 证据审计
```text
EA = EvidenceAudit(V, S, RCG)
```

检查：
- 反思中提到的时间段/区域/对象是否真实存在
- 所说“没看清/没跟上/遮挡/断轨”是否有证据

---

## Step 5: 因果合理性审计
```text
CA = CausalAudit(task, S, Y, RCG)
```

看：
- 错因链是否合理
- proposed intervention 是否匹配错因
- 这份 reflection 是具体指向性，还是套话

---

## Step 6: 生成可验证 intervention 候选
```text
I_candidates = GenerateInterventions(RCG, EA, CA, budget)
```

例如：
- inspect frames 42-48
- zoom region hand-cup
- rerun object contact detector
- extend context to 30 frames before/after
- abstain if no decisive evidence

---

## Step 7: 按“鉴别力优先”选择 intervention
```text
I_star = SelectIntervention(I_candidates, criterion="max_disambiguation_per_cost")
```

这里不是选“最可能提升答案”的，而是优先选：
- 最能验证当前 reflection 是否靠谱
- 最能区分 competing explanations 的动作

例如两种 competing explanations：
- A: 接触没看清
- B: 事件边界切错

那优先做能区分 A/B 的 intervention。

---

## Step 8: 执行 intervention 并重评估
```text
S' = ApplyIntervention(V, S, I_star)
Y' = ReReason(S', q)
R' = Reflect(V, S', Y', trace')
```

重新生成：
- 新状态
- 新输出
- 新 reflection

---

## Step 9: 比较前后变化，估计 reflection 可信度
模块看：

- 干预是否改善原 claim？
- 改善是否符合 reflection 的预期？
- 错因归因是否被支持？
- 反思是否变得更稳定、更具体？

形成：

```text
A = MetaAudit(EA, CA, delta(Y,Y'), delta(R,R'), intervention_outcome)
```

---

## Step 10: 输出决策
根据 `A` 决定：

- 接受修正答案
- 再做一轮 targeted intervention
- 触发 stronger verifier
- 保留不确定并 abstain
- 若是 embodied task，则重新规划动作

---

---
# 6. “证自证分”的关键评分维度
---

不用 loss，但需要定义运行时指标/分数。

建议至少有以下几个 runtime score。

---

## 6.1 Reflection Groundedness Score (RGS)
反思内容与感知证据的一致程度。

高分条件：
- 提到的对象、时间段、区域都存在
- 所述不确定性与实际 uncertainty map 一致
- unsupported claim 真 unsupported

---

## 6.2 Attribution Validity Score (AVS)
错因归因的可信度。

高分条件：
- 所归因错误路径与任务结构吻合
- 能解释当前错误模式
- 不与已有感知证据冲突

---

## 6.3 Intervention Diagnostic Value (IDV)
某 intervention 对验证 reflection 真伪的鉴别力。

高分 intervention：
- 能强区分 competing explanations
- 成本低
- 对当前高风险 claim 有帮助

---

## 6.4 Reflection Predictive Reliability (RPR)
反思对后续修正结果的预测有效性。

例如：
- 它说 zoom 会有用，zoom 后确有用
- 它说 tracking 重跑会修复，确实修复

---

## 6.5 Meta-Trust Score (MTS)
综合形成“是否相信这次 reflection”。

最终：
- 高 MTS → 接受 self-correction
- 中 MTS → 小成本验证后再决定
- 低 MTS → 不信其自省，切换其他 verifier / 重新感知 / abstain

---

---
# 7. 关键算法机制：竞争性反思假设
---

我建议不要只让系统审计一份 reflection，  
而是显式构造多个**竞争性反思假设**，这会更稳。

---

## 7.1 为什么
很多时候一份 reflection 是后验编造。  
所以你应该至少保留几类竞争解释：

- H1: perception blur
- H2: tracking error
- H3: temporal boundary error
- H4: language prior hallucination
- H5: missing long-range context

然后证自证分模块做的是：
- 比较这些反思假设
- 选择最能区分它们的 intervention

这比“相信第一份 reflection”要强得多。

---

## 7.2 具体流程
1. 由 reflector 生成 top-k reflection hypotheses
2. 审计各自 groundedness 和 plausibility
3. 找出最难分的两个/三个假设
4. 选一个最便宜但最有区分力的 intervention
5. 用 intervention outcome 更新 hypothesis ranking

这几乎就是一个**元层诊断过程**。

---

---
# 8. 面向 video 的特殊设计点
---

video agent 和图像 agent 不一样，证自证分模块必须利用时序。

---

## 8.1 时间局部化优先
反思必须尽量指出：
- 哪个时间段有问题
- 哪个 event boundary 可疑
- 哪段轨迹不可靠

否则没法验证。

---

## 8.2 事件级而不是帧级
很多错误不是单帧看错，而是事件关系错。

所以表征要支持：
- event candidate graph
- event boundary confidence
- inter-event causal links

reflection 也要能说：
- “动作 A 到 B 的转折边界不清”
而不是只说“视频太模糊”

---

## 8.3 利用多尺度时间窗
证自证分模块在 intervention 上要支持：
- 微观：重看 5 帧
- 中观：重看 1-2 秒局部
- 宏观：拉长前后上下文

很多错因在不同时间尺度上才显现。

---

## 8.4 行动后果验证
如果是 embodied video agent，可以用动作结果反证 reflection：

- 反思说“杯子已被拿起”
- 计划基于此继续行动
- 若后续动作失败，说明此前状态判断和/或 reflection 可疑

即 action outcome 是证自证分的重要外部证据。

---

---

# 9. 一个简化版伪代码

```python
def meta_reflective_audit(video, query, state_repr, answer, reflection, history, budget):

    # 1. parse reflection into structured graph
    rcg = parse_reflection(reflection, answer)

    # 2. evidence audit
    ea = evidence_audit(video, state_repr, rcg)

    # 3. causal plausibility audit
    ca = causal_audit(query, state_repr, answer, rcg)

    # 4. build competing reflection hypotheses
    hypotheses = build_competing_hypotheses(reflection, rcg, ea, ca, history)

    # 5. score hypotheses
    hyp_scores = score_hypotheses(hypotheses, ea, ca, history)

    # 6. generate candidate interventions
    interventions = generate_interventions(hypotheses, state_repr, budget)

    # 7. choose intervention by diagnostic value / cost
    selected = select_intervention(interventions, hypotheses, hyp_scores, budget)

    # 8. if no useful intervention, return trust and decision directly
    if selected is None:
        mts = aggregate_meta_trust(ea, ca, hyp_scores, history)
        decision = decide_without_intervention(mts, hyp_scores)
        return {
            "meta_trust": mts,
            "hypotheses": hyp_scores,
            "selected_intervention": None,
            "decision": decision
        }

    # 9. apply selected intervention
    new_state_repr = apply_intervention(video, state_repr, selected)

    # 10. rerun localized reasoning / replanning
    new_answer = rerun_reasoning(query, new_state_repr, answer, selected)

    # 11. rerun reflection
    new_reflection = rerun_reflection(video, query, new_state_repr, new_answer, answer, reflection)

    # 12. compare delta
    delta = compare_before_after(
        old_state=state_repr,
        new_state=new_state_repr,
        old_answer=answer,
        new_answer=new_answer,
        old_reflection=reflection,
        new_reflection=new_reflection,
        selected_intervention=selected
    )

    # 13. assess predictive validity of old reflection
    pv = predictive_validity_assessment(
        old_reflection=reflection,
        rcg=rcg,
        selected_intervention=selected,
        delta=delta
    )

    # 14. recompute meta trust
    mts = aggregate_meta_trust(ea, ca, hyp_scores, history, predictive_validity=pv, delta=delta)

    # 15. final decision
    decision = final_decision(
        meta_trust=mts,
        old_answer=answer,
        new_answer=new_answer,
        old_reflection=reflection,
        new_reflection=new_reflection,
        delta=delta,
        budget=budget
    )

    return {
        "meta_trust": mts,
        "hypotheses": hyp_scores,
        "selected_intervention": selected,
        "delta": delta,
        "predictive_validity": pv,
        "new_answer": new_answer,
        "new_reflection": new_reflection,
        "decision": decision
    }
```

---

# 10. 关键子模块展开

下面把每个函数写成比较工程化的定义。

---

## 10.1 `parse_reflection`

目标：把 reflection 变成可验证的结构化对象。

### 输入
- `reflection`
- `answer`

### 输出
- `rcg = reflection_claim_graph`

### 结构建议

```python
rcg = {
    "target_claims": [
        {
            "claim_id": "c2",
            "claim_text": "person picks up the cup",
            "status": "possibly_wrong"
        }
    ],
    "attribution_claims": [
        {
            "type": "contact_ambiguity",
            "span": [42, 48],
            "objects": ["hand#1", "cup#3"],
            "confidence": 0.77
        }
    ],
    "uncertainty_claims": [
        {
            "type": "low_resolution",
            "span": [42, 48],
            "region": [x1, y1, x2, y2]
        }
    ],
    "intervention_claims": [
        {
            "type": "zoom_region",
            "span": [42, 48],
            "region": [x1, y1, x2, y2]
        },
        {
            "type": "rerun_contact_detector",
            "span": [42, 48],
            "objects": ["hand#1", "cup#3"]
        }
    ],
    "causal_links": [
        ("low_resolution", "contact_ambiguity"),
        ("contact_ambiguity", "wrong_pickup_claim")
    ]
}
```

### 要点
如果 reflection 没有结构化输出，运行时可以让 LLM 做一次强约束解析：
- 只允许从固定 ontology 中选 error type
- 每个归因必须绑定时间段
- 每个修正建议必须绑定对象/区域/工具

否则没法审计。

---

## 10.2 `evidence_audit`

目标：验证 reflection 中提到的“我为什么错”是否有底层证据支持。

### 输入
- `video`
- `state_repr`
- `rcg`

### 输出
- 每条 reflection claim 的支持度

### 检查项

#### A. 时间定位检查
reflection 提的 span 是否合理？
- 是否落在视频范围内
- 是否与原 claim 所在时间段重叠
- 是否是高不确定段

#### B. 区域证据检查
如果说“右下角看不清”
- 该 region 是否真有低分辨率/遮挡/运动模糊
- 是否目标对象出现在该 region

#### C. 轨迹证据检查
如果说“track 断了”
- 轨迹置信度是否下降
- IoU/association 是否不稳
- 是否在该段发生 occlusion

#### D. 事件证据检查
如果说“事件边界不清”
- event segmentation confidence 是否低
- 前后事件是否强混叠

### 示例输出

```python
ea = {
    "claim_support": {
        "attr_1": {
            "support_score": 0.86,
            "matched_signals": ["low_contact_conf", "track_fragmentation"],
            "conflicts": []
        },
        "unc_1": {
            "support_score": 0.74,
            "matched_signals": ["motion_blur_high"],
            "conflicts": []
        },
        "int_1": {
            "support_score": 0.69,
            "matched_signals": ["small_object_region"],
            "conflicts": []
        }
    },
    "overall_groundedness": 0.79
}
```

---

## 10.3 `causal_audit`

目标：判断 reflection 的错因链是否“讲得通”。

### 输入
- `query`
- `state_repr`
- `answer`
- `rcg`

### 输出
- 归因路径合理性
- 修正建议与归因是否匹配

### 例子
如果任务是识别“是否拿起杯子”，则：
- “接触关系模糊 → 抓取判断错” 合理
- “颜色不确定 → 抓取判断错” 除非任务涉及颜色，不然不太合理

### 检查维度
1. **Task relevance**
2. **Causal locality**
3. **Intervention-match**
4. **Over-generalization penalty**

### 示例输出

```python
ca = {
    "attribution_validity": 0.81,
    "intervention_match": {
        "zoom_region": 0.84,
        "rerun_contact_detector": 0.91,
        "ask_for_more_context": 0.33
    },
    "causal_conflicts": [],
    "genericity_penalty": 0.12
}
```

---

## 10.4 `build_competing_hypotheses`

目标：避免只信一份 reflection，构造几个竞争解释。

### 输入
- 原 reflection
- `rcg`
- `ea`
- `ca`
- `history`

### 输出
- `hypotheses`

### Hypothesis 形式

```python
hypotheses = [
    {
        "id": "H1",
        "type": "contact_ambiguity_due_to_low_res",
        "explains": ["c2"],
        "required_evidence": ["contact_conf_low", "blur_high"],
        "best_interventions": ["zoom_region", "rerun_contact_detector"]
    },
    {
        "id": "H2",
        "type": "temporal_boundary_error",
        "explains": ["c2"],
        "required_evidence": ["boundary_conf_low"],
        "best_interventions": ["expand_temporal_window", "rerun_event_segmentation"]
    },
    {
        "id": "H3",
        "type": "language_prior_hallucination",
        "explains": ["c2"],
        "required_evidence": ["weak_visual_support", "strong_prior_pattern"],
        "best_interventions": ["evidence_only_rereason", "claim_grounding_check"]
    }
]
```

### 来源
- 一条来自原 reflection
- 几条来自系统默认 error library
- 几条来自历史最常见失败模式

这样即使原 reflection 是假反思，也能被对照出来。

---

## 10.5 `score_hypotheses`

目标：给竞争性反思假设打分。

### 评分组成
可由以下因素组合：
- evidence support
- causal plausibility
- historical reliability
- conflict with current state
- specificity

### 输出示意

```python
hyp_scores = {
    "H1": {
        "posterior": 0.48,
        "evidence_support": 0.82,
        "causal_plausibility": 0.87,
        "historical_reliability": 0.61
    },
    "H2": {
        "posterior": 0.31,
        "evidence_support": 0.55,
        "causal_plausibility": 0.78,
        "historical_reliability": 0.64
    },
    "H3": {
        "posterior": 0.21,
        "evidence_support": 0.49,
        "causal_plausibility": 0.58,
        "historical_reliability": 0.40
    }
}
```

---

## 10.6 `generate_interventions`

目标：生成可用来验证/修复 reflection 的动作。

### intervention 类型库

#### 感知类
- zoom region
- sample dense frames in span
- rerun detector
- rerun tracker
- rerun OCR
- rerun contact estimator
- multi-scale temporal encoding

#### 推理类
- evidence-only reasoning
- remove language prior prompt
- enforce claim grounding
- local event re-segmentation

#### 交互类（如果是 embodied）
- move camera / request next view
- pause and inspect
- retry manipulation with sensing

#### 决策类
- abstain
- ask clarifying question
- escalate to stronger verifier

### 输出示例

```python
interventions = [
    {
        "id": "I1",
        "type": "zoom_region",
        "target_hypotheses": ["H1"],
        "cost": 1.0,
        "diagnostic_value": 0.72
    },
    {
        "id": "I2",
        "type": "expand_temporal_window",
        "target_hypotheses": ["H2"],
        "cost": 1.4,
        "diagnostic_value": 0.65
    },
    {
        "id": "I3",
        "type": "evidence_only_rereason",
        "target_hypotheses": ["H3"],
        "cost": 0.6,
        "diagnostic_value": 0.54
    },
    {
        "id": "I4",
        "type": "rerun_contact_detector",
        "target_hypotheses": ["H1", "H2"],
        "cost": 1.2,
        "diagnostic_value": 0.81
    }
]
```

---

## 10.7 `select_intervention`

目标：选最“值”的 intervention。

关键原则：

> 不是简单选最可能提升准确率的，而是选最能验证 reflection 真伪、区分 competing hypotheses 的。

### 选择标准
- 高 diagnostic value
- 低 cost
- 高 expected information gain
- 高 claim risk coverage
- 在预算内

### 简化决策规则
优先选：
1. 能区分当前 top-2 hypotheses 的
2. 成本最低
3. 对高风险 claim 有直接作用

---

## 10.8 `apply_intervention`

目标：真正执行局部修复或验证。

### 例子
- `zoom_region`：重采样该时间段局部高分辨 token
- `expand_temporal_window`：把原 16 帧窗口扩展到 64 帧
- `rerun_contact_detector`：在指定轨迹上重跑接触检测
- `evidence_only_rereason`：禁用常识补全，只允许引用 visual support

输出：
- 更新后的 `state_repr'`

---

## 10.9 `rerun_reasoning`

目标：在新的状态上做局部重推理，而不是整个系统从头跑。

### 例子
只针对受影响 claims 重算：
- `c2 = person picks up cup`
- `c3 = hand leaves table`

避免系统全局漂移。

---

## 10.10 `compare_before_after`

目标：比较 intervention 前后的关键变化。

### 比较项
1. answer 有没有变化
2. claim 置信度如何变化
3. support evidence 是否增强
4. reflection 是否变得更具体/更稳定
5. competing hypotheses 的排序是否改变

### 输出示意

```python
delta = {
    "answer_changed": True,
    "claim_updates": {
        "c2": {
            "old_conf": 0.74,
            "new_conf": 0.41,
            "status": "downgraded"
        }
    },
    "evidence_updates": {
        "contact_support": {
            "old": 0.32,
            "new": 0.61
        }
    },
    "reflection_shift": {
        "more_specific": True,
        "attribution_changed": False
    }
}
```

---

## 10.11 `predictive_validity_assessment`

这一步是证自证分最像“证”的地方。

目标：判断旧 reflection 的预测是否被 intervention 结果支持。

### 如果旧 reflection 说：
- “放大 hand-cup 区域会澄清接触关系”

那 intervention 后应看到：
- contact evidence 明显增强
- 原 claim 变得更确定，或者被修正
- 不应完全无变化

### 输出示意

```python
pv = {
    "supported": True,
    "support_degree": 0.83,
    "matched_expectations": [
        "contact evidence increased",
        "target claim confidence updated"
    ],
    "failed_expectations": []
}
```

如果 reflection 预言的修复没发生，则：
- reflection 的可信度下降
- 原归因可能是假反思

---

## 10.12 `aggregate_meta_trust`

目标：综合所有信号得到最终 Meta-Trust Score。

### 输入来源
- `ea.overall_groundedness`
- `ca.attribution_validity`
- `hyp_scores`
- `pv.support_degree`
- `history` 中类似反思的成功率
- 是否存在强冲突

### 输出

```python
mts = {
    "overall": 0.76,
    "trust_reflection_text": 0.73,
    "trust_attribution": 0.81,
    "trust_fix_proposal": 0.79,
    "risk_of_false_reflection": 0.18
}
```

---

## 10.13 `final_decision`

根据 meta-trust 做最后决策。

### 可能决策
- `accept_revision`
- `accept_original_but_mark_uncertain`
- `run_another_targeted_intervention`
- `switch_to_alternate_verifier`
- `abstain`
- `request_more_video_context`
- `replan_action`

### 简化策略
- 高 trust + 有效修复 → 接受修正
- 中 trust + 高风险任务 → 再验证一次
- 低 trust → 不信 reflection，改走别的校验路径
- 极低 trust + 证据不足 → abstain / defer

---

# 11. 面向 video agent 的状态表示建议

如果你真要实现，最容易失败的地方是：  
`state_repr` 太弱，导致证自证分模块什么都审计不了。

所以建议 `state_repr` 最低至少包含这几层：

---

## 11.1 时序对象层
```python
object_tracks = [
    {
        "obj_id": "cup#3",
        "boxes": [...],
        "track_conf": [...],
        "visibility": [...],
        "attributes": {...}
    }
]
```

---

## 11.2 事件候选层
```python
event_candidates = [
    {
        "event_id": "e1",
        "type": "pick_up",
        "span": [42, 58],
        "participants": ["hand#1", "cup#3"],
        "conf": 0.62
    }
]
```

---

## 11.3 不确定性层
```python
uncertainty_map = {
    "frame_level": [...],
    "track_level": {...},
    "event_level": {...},
    "region_level": {...}
}
```

---

## 11.4 证据追踪层
```python
support_trace = {
    "claim_to_evidence": {
        "c2": {
            "frames": [42, 45, 47],
            "tracks": ["hand#1", "cup#3"],
            "detectors": ["contact_estimator"]
        }
    }
}
```

没有这层，reflection 很容易胡说，auditor 也很难查。

---

继续。下面把这套“证自证分模块”进一步收束成一个**更完整的 video agent runtime 方案**：包括状态机、接口、一次完整运行示例、失败模式与保护策略、以及如何接到现有 video agent 上。

---

# 12. 运行时状态机

把它做成状态机会比较稳，不然很容易无限反思循环。

## 12.1 状态定义

建议定义以下运行状态：

- `S0_INIT`
  - 初始感知与回答
- `S1_REFLECT`
  - 生成自证分：反思、自评、归因、修正建议
- `S2_AUDIT`
  - 证自证分：审计反思是否可信
- `S3_INTERVENE`
  - 执行最小鉴别性干预
- `S4_REEVAL`
  - 局部重推理与再反思
- `S5_DECIDE`
  - 输出最终决策
- `S_ABORT`
  - 放弃继续验证，返回不确定/请求更多信息

---

## 12.2 状态转移规则

### 初始
`S0_INIT -> S1_REFLECT`

### 常规
`S1_REFLECT -> S2_AUDIT`

### 审计后
- 如果 `meta_trust` 很高，且风险低  
  `S2_AUDIT -> S5_DECIDE`
- 如果 `meta_trust` 中等，且存在高价值 intervention  
  `S2_AUDIT -> S3_INTERVENE`
- 如果 `meta_trust` 很低，但还有替代路径  
  `S2_AUDIT -> S3_INTERVENE` 或切换 alternate verifier
- 如果预算不足 / 证据不足 / 不值得继续  
  `S2_AUDIT -> S_ABORT`

### 干预后
`S3_INTERVENE -> S4_REEVAL -> S2_AUDIT`  
形成一个有限回路。

---

## 12.3 终止条件

必须加，不然 agent 会一直“反思-再反思”。

建议终止条件：

1. 达到最大审计轮数 `K_max`
2. 预算耗尽
3. 新 intervention 的诊断价值低于阈值
4. 连续两轮 `meta_trust` 提升很小
5. 高风险 claim 已经足够稳定
6. 任务时延要求不允许继续

---

# 13. 推荐的模块接口

这里给你一套较清晰的工程接口定义。

---

## 13.1 主 agent 输出接口

```python
AgentOutput = {
    "answer": answer,
    "claims": claims,
    "state_repr": state_repr,
    "trace": trace
}
```

其中：

### `claims`
```python
claims = [
    {
        "claim_id": "c1",
        "type": "event",
        "text": "person picks up the cup",
        "span": [42, 58],
        "confidence": 0.74,
        "support_refs": ["track:hand#1", "track:cup#3", "event:e1"]
    }
]
```

### `trace`
```python
trace = {
    "used_frames": [36, 40, 44, 48, 52, 56],
    "attention_regions": ...,
    "retrieved_memories": ...,
    "tool_calls": ...,
    "reasoning_summary": ...
}
```

---

## 13.2 Reflection 接口

```python
ReflectionOutput = {
    "overall_self_confidence": 0.63,
    "claim_reviews": [
        {
            "claim_id": "c1",
            "status": "possibly_wrong",
            "attribution": "contact_ambiguity",
            "span": [42, 48],
            "objects": ["hand#1", "cup#3"],
            "evidence_gap": "contact relationship unclear",
            "proposed_fix": ["zoom_region", "rerun_contact_detector"],
            "expected_effect": "disambiguate approach vs pickup"
        }
    ],
    "global_notes": ...,
    "raw_text": ...
}
```

---

## 13.3 Meta-Audit 输出接口

```python
MetaAuditOutput = {
    "meta_trust": {
        "overall": 0.76,
        "groundedness": 0.79,
        "attribution_validity": 0.81,
        "fix_validity": 0.78,
        "predictive_validity": 0.73
    },
    "hypotheses": [...],
    "recommended_action": "run_targeted_intervention",
    "selected_intervention": {...},
    "audit_report": {...}
}
```

---

# 14. 运行时主算法，整理成一步一步版本

这里给一个更适合实现文档的版本。

---

## Algorithm: Meta-Reflective Auditing for Video Agents

### 输入
- 视频 `V`
- 任务/问题 `Q`
- 主 agent `A`
- reflection 模块 `R`
- meta-auditor `M`
- intervention executor `I`
- 最大轮数 `K`
- 预算 `B`

### 输出
- 最终答案/动作 `Y*`
- 最终反思 `R*`
- 审计报告 `Audit*`

---

### Step 1. 初始感知与推理
主 agent 对视频进行感知和任务求解：

```python
S0 = A.perceive(V, Q)
Y0 = A.reason(S0, Q)
```

输出：
- 世界状态 `S0`
- 初始 claims `Y0.claims`

---

### Step 2. 生成自证分
反思模块对当前结果做自省：

```python
R0 = R.reflect(V, Q, S0, Y0)
```

要求给出：
- 哪些 claim 可能错
- 错因是什么
- 证据缺口在哪
- 应做什么干预来验证/修复

---

### Step 3. 证自证分审计
Meta-auditor 对 `R0` 做二阶审计：

```python
A0 = M.audit(V, Q, S0, Y0, R0)
```

输出：
- 反思可信度
- 竞争假设
- 推荐干预

---

### Step 4. 决策是否直接接受反思
如果：
- `A0.meta_trust.overall` 足够高
- 任务风险低
- 或 reflection 已建议 abstain 且审计支持

则可直接进入最终决策：

```python
if accept_without_intervention(A0):
    return finalize(Y0, R0, A0)
```

---

### Step 5. 选择最小鉴别性干预
若需要验证，则选一个 cost-effective 且 diagnostic 的 intervention：

```python
I0 = A0.selected_intervention
```

---

### Step 6. 执行局部干预
对目标时间段、区域或事件局部重感知：

```python
S1 = I.apply(V, S0, I0)
Y1 = A.local_reason(S1, Q, affected_claims=R0.target_claims)
R1 = R.reflect(V, Q, S1, Y1, prev_reflection=R0)
```

---

### Step 7. 比较干预前后
Meta-auditor 检查：

- 原反思预测的变化是否发生
- claim 是否被修正/确认
- 证据是否增强
- 反思是否变得更具体和一致

```python
A1 = M.audit_with_delta(V, Q, S0, Y0, R0, S1, Y1, R1, I0)
```

---

### Step 8. 决定是否继续下一轮
若：
- meta-trust 仍不够高
- 但存在高价值下一步 intervention
- 且预算允许

则再跑一轮。

否则终止并输出。

---

# 15. 一次完整示例：视频中“是否拿起杯子”

这里给一个具体例子，方便你感受模块如何工作。

---

## 15.1 任务
输入一个视频，问题：

> “视频中的人是否拿起了桌上的杯子？”

---

## 15.2 初始 agent 输出
主 agent 输出：

```text
Answer: Yes
Claim c1: person picks up the cup during 42-58
Confidence: 0.74
```

支持来自：
- hand 轨迹接近 cup
- cup 在后续帧位置改变

---

## 15.3 Reflection 输出
reflection 说：

```text
我对 c1 不是完全确定。
可能错误原因是 42-48 帧中手和杯子的接触关系不清，
我可能把“靠近”误看成“拿起”。
建议放大该时间段手-杯区域，并重跑接触检测。
```

---

## 15.4 Meta-auditor 审计
证自证分模块检查：

### 证据层面
- 42-48 帧确实 motion blur 较高
- hand/cup track 在 45 帧附近置信下降
- contact detector 原输出置信低

### 因果层面
- 对“是否拿起”的判断，接触关系是关键因子
- “zoom + rerun contact detector”是合理修复

### 竞争假设
- H1: 接触关系模糊
- H2: 事件边界错，把后续杯子位移误当成拿起
- H3: 语言先验投射，“伸手靠近桌上物品”常被误判为拿起

当前排序：
- H1 最高
- H2 次之
- H3 较低

于是选择 intervention：
- `zoom_region(frames 42-48, hand-cup region)`
- `rerun_contact_detector`

---

## 15.5 干预后
局部重感知后发现：

- 手确实接近杯子，但没有稳定抓取闭合
- 杯子位置变化来自镜头角度和后续桌面滑动，不是手拿起

新答案：

```text
Answer: No / Uncertain leaning No
Claim c1 confidence from 0.74 -> 0.38
```

新 reflection：

```text
原判断主要受接近动作诱导，
在关键帧中缺少明确抓取证据。
之前的“拿起”判断不成立。
```

---

## 15.6 证自证分结论
Meta-auditor 判断：

- 原 reflection 有较强 predictive validity
- 原归因基本正确
- 修正建议有效
- 可以接受修订答案

最终输出：
- 改答案
- 附带可信审计报告

这就是“证自证分”的效果：  
不是单纯反思，而是**验证反思本身是否抓住了真实问题。**

---

# 16. 如果是 embodied / interactive video agent，怎么改？

如果 agent 不只是看视频回答，而是边看边行动，那么证自证分会更强，因为可以借动作结果来反证。

---

## 16.1 增加 action outcome 作为审计证据

例如机器人判断：
- “杯子已经被拿起了”

然后基于此做下一步动作：
- 移动到放置位置

如果动作失败，说明：
- 世界状态估计错
- 或反思不可靠

所以审计输入增加：

```python
action_outcome = {
    "success": False,
    "failure_mode": "object_not_in_gripper"
}
```

证自证分模块会更新：
- 对之前 self-assessment 的 trust
- 对相关 error type 的 historical reputation

---

## 16.2 新增 intervention 类型
interactive 场景下，intervention 不只是算子，还包括主动观察：

- move camera closer
- inspect from another viewpoint
- pause before acting
- tactile check
- re-grasp trial
- ask user confirmation

这样“证自证分”更像真正的元认知控制器。

---

# 17. 历史记忆：Reflection Reputation Memory 的运行时用法

这是很重要的一层，否则每次都是一次性判断。

---

## 17.1 存什么
建议对每次审计存一个 compact record：

```python
memory_item = {
    "task_type": "pickup_detection",
    "video_context_signature": ...,
    "reflection_type": "contact_ambiguity",
    "selected_fix": "zoom+contact_rerun",
    "pre_audit_trust": 0.68,
    "post_validation_result": "supported",
    "improvement": 0.31,
    "final_outcome_corrected": True
}
```

---

## 17.2 怎么用
当新任务来时，根据：
- 任务类型
- 当前 claim 类型
- 视频特征
- error type

检索类似历史案例，提供：

- 这类 reflection 过去是否靠谱
- 这类 intervention 平均收益如何
- 哪种 competing hypothesis 最常被误选

这会让 meta-auditor 不只是当下判断，而是有“元经验”。

---

# 18. 失败模式与保护策略

这个模块也会出错，所以必须设计保护。

---

## 18.1 失败模式 1：反思模板化
问题：
reflection 永远说：
- “信息不足，建议查看更多帧”

### 保护
- 对泛化过强的 reflection 加 genericity penalty
- 要求反思必须绑定具体时间段、对象、claim
- 如果多次都是泛化模板，降低其信誉

---

## 18.2 失败模式 2：审计器和反思器同偏
问题：
都来自同一个大模型家族，互相附和。

### 保护
- 审计器尽量用异质工具
  - tracker / OCR / contact detector / event segmenter
- 审计基于 structured evidence，不只看文本
- 关键任务引入外部 verifier

---

## 18.3 失败模式 3：无限验证循环
问题：
不断说“再看一点就更清楚了”。

### 保护
- 严格预算
- 最大轮数
- marginal gain 小于阈值就停止
- 对高成本低诊断值 intervention 直接拒绝

---

## 18.4 失败模式 4：把真正的感知缺陷误归因为高层推理问题
问题：
其实是没看到，reflection 却说是 reasoning ambiguity。

### 保护
- 优先做 perception-level audit
- 先检查 grounding，再接受抽象解释
- “无证据高层解释”一律降权

---

## 18.5 失败模式 5：修复有效但归因错误
问题：
zoom 后答案变了，但不是 reflection 说的那个原因。

### 保护
- 不只看“变好了没”，还看“是否按预期方式变好”
- 做 expectation matching
- 必要时保留多个 competing hypotheses

---

# 19. 一个更紧凑的最小可行实现版本

如果你现在就要做一个 prototype，我建议不要一开始做太重。

## 19.1 MVP 架构
只保留 5 个核心件：

1. **Video Agent**
   - 输出 answer + claim + support span

2. **Reflector**
   - 输出 target claim + error type + time span + fix proposal

3. **Evidence Auditor**
   - 检查反思是否 grounded

4. **Intervention Executor**
   - 只支持少量 intervention：
     - zoom local span
     - extend temporal window
     - rerun detector/tracker
     - evidence-only reasoning

5. **Meta Decision**
   - 根据前后变化决定是否信 reflection

---

## 19.2 MVP 支持的 error taxonomy
先不要搞太多类型，控制在 5 类：

- visual ambiguity
- temporal boundary error
- tracking failure
- OCR ambiguity
- language prior hallucination

---

## 19.3 MVP 支持的 intervention
控制在 4 类：

- local zoom
- temporal expand
- rerun tracker/detector
- evidence-only rereason

这样足以证明“证自证分”的概念。

---

# 20. 建议的系统提示词/协议化输出

为了实现稳定，reflection 最好强制结构化。比如要求输出 JSON：

```json
{
  "overall_self_confidence": 0.63,
  "claim_reviews": [
    {
      "claim_id": "c1",
      "status": "possibly_wrong",
      "error_type": "visual_ambiguity",
      "time_span": [42, 48],
      "objects": ["hand#1", "cup#3"],
      "evidence_gap": "contact relation unclear",
      "proposed_fix": [
        {"type": "zoom_region", "region_ref": "hand_cup_region"},
        {"type": "rerun_contact_detector", "objects": ["hand#1", "cup#3"]}
      ],
      "expected_change": "pickup confidence should decrease if no stable grasp is detected"
    }
  ]
}
```

注意最后这个字段很重要：

- `expected_change`

因为证自证分模块就是靠这个来验证：
**你不仅要说怎么改，还要说改后应该看到什么。**

---

# 21. 这套模块最本质的创新点

从方法上看，它和普通 reflection 的区别是：

## 普通 reflection
- 给出错误原因
- 给出修复建议

## 证自证分模块
- 把错误原因当成**可检验假设**
- 把修复建议当成**诊断性实验**
- 根据实验结果更新“是否相信这份反思”

这就是本质差异。

---

# 22. 给它一个正式名字

你可以考虑几个名字：

- **Meta-Reflective Auditor**
- **Second-Order Introspection Verifier**
- **Reflective Reliability Controller**
- **Self-Assessment Validation Module**
- **CZZZF Module**（不建议对外这么叫）

如果保留一点唯识 flavor，但又不太玄，可以叫：

- **Second-Order Reflective Verification**
- **Meta-Reflective Grounding and Validation**

我个人偏好：
> **Meta-Reflective Auditor for Video Agents**

---

# 23. 最终简版算法摘要

最后我把它压成一段论文/proposal 风格的摘要描述：

> We introduce a meta-reflective auditor for video agents, a second-order module that does not directly verify task outputs, but instead validates the agent’s own self-assessment, error attribution, and proposed correction strategy. Given the initial video-derived claims and a structured reflection, the auditor evaluates whether the reflection is grounded in observable evidence, causally plausible with respect to the task, and predictive of effective interventions. It then selects minimal diagnostic interventions—such as localized zoom-in, temporal window expansion, detector reruns, or evidence-only re-reasoning—to test the reflection itself. The resulting outcome is used to update a meta-trust score over the reflection, which governs whether the system accepts the self-correction, requests additional perception, switches verification pathways, or abstains. This enables video agents to not only reflect, but also to assess whether their own reflection deserves trust.