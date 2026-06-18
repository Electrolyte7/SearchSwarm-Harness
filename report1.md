# SearchSwarm Harness Patch 阶段总结

## 1. Git 状态

- 当前分支：`main`
- Commit hash：提交后见最终 git commit；提交前基线为 `f7a6580`
- 是否成功 commit：是，本报告随 patch 提交
- 是否成功 push：提交后尝试 push；最终状态见对话总结
- git status 是否 clean：提交前仍有未纳入提交的本地文件；本次提交只纳入代码、测试、报告和小型 eval subset

## 2. 修改文件列表

| 文件 | 改动 |
|---|---|
| `tool_sub_agent.py` | Patch v1：budget-aware early stop、duplicate brief filter、low-quality report flag |
| `patch_v2.py` | Patch v2/v2.1：candidate ledger、final verifier、relation-aware scoring |
| `react_agent.py` | 接入 v2 ledger/verifier/router、main-agent early finalize、post-verifier sanitizer |
| `scripts/validate_smoke_run.py` | summary 聚合 v1/v2 patch 字段 |
| `scripts/run_benchmark_variant.sh` | run_config 记录 patch 环境变量，validation skipped 路径补 summary |
| `tests/test_smoke_regressions.py` | 增加 v1/v2/v2.1 mock 和 regression tests |
| `eval_data/benchmark/patch_v2/browsecomp_item0_sanity.jsonl` | 小型 targeted sanity 数据集：BrowseComp item 0 |
| `eval_data/benchmark/patch_v2/browsecomp_item18_td.jsonl` | 小型 targeted validation 数据集：BrowseComp item 18 |

## 3. Patch 模块总结

### Patch v1: Budget-aware Delegation

- sub-agent budget-aware early stop
- duplicate brief filter
- low-quality report flag
- summary 字段扩展

效果：
- 降低 sub-agent max_calls；
- 减少无效 fallback；
- 给 main agent 提供 report quality 信号。

### Patch v2: Adaptive Evidence-Grounded Harness

- Candidate / Evidence Ledger
- Constraint-aware Final Verifier
- Adaptive Delegation Router
- Main-agent Early Finalize
- verifier 前后双层 final sanitizer

效果：
- 显式维护 candidate/evidence；
- 在 final answer 前做 evidence-aware verification；
- 减少错误候选固化；
- 避免 verifier 输出长解释、DSML、tool-call 或 JSON-like final。

### Patch v2 Conservative Tightening

- 默认保留 draft answer；
- 过滤 generic/source/tool-artifact candidate；
- 防止 `ambiguous`、`Brock News`、`bootstrap smoke sub-agent report` 等 noisy candidate 覆盖 final；
- replacement 必须通过强 gate。

效果：
- item 0 sanity 恢复正确；
- verifier 不再把原本正确答案改坏。

### Patch v2.1 Relation-aware Verifier

- relation candidate extraction；
- `Brought to you by X` / `program through X` 等关系抽取；
- `answer_role` / `target_role`；
- role-aware scoring；
- role-mismatch replacement exception。

效果：
- 区分正文相关实体、source name、sponsor/provider/program；
- 修复 item 18 中 `peaksaver` / `Brock News` 误导问题。

## 4. 离线测试结果

- py_compile：passed
- bash -n：passed
- unittest：passed
- 当前测试数：`Ran 50 tests ... OK`
- 覆盖行为：
  - duplicate filter
  - early stop
  - report quality
  - final verifier
  - conservative gate
  - relation extraction
  - post-verifier sanitizer

## 5. 已有真实 API targeted validation

### Patch v1

- item 13：
  - baseline failed；
  - patched still failed；
  - 但 subagent max_calls 从 2 降到 0，early_stop 生效。

- item 18：
  - v1 曾从 `peaksaver` 修正到接近 `TD Insurance Meloche Monnex Program`；
  - 说明 early stop / low-quality report 有方向性收益，但当时仍不稳定。

### Patch v2 pre-tightening

- item 0 曾被错误改成 `ambiguous`；
- item 18 曾输出 `Brock News`；
- 说明初版 verifier 过度相信 noisy ledger。

### Patch v2 Conservative Tightening

- item 0 sanity 通过：
  - gold: `The Cuban Missile Crisis, 1990 May 08`
  - prediction: `The Cuban Missile Crisis, May 8, 1990`
  - verifier changed answer: false
  - 说明 conservative verifier 不再改坏正确答案。

### Patch v2.1 Relation-aware Validation

- item 18 成功：
  - gold: `TD Insurance Meloche Monnex`
  - prediction: `TD Insurance Meloche Monnex`
  - likely_correct: true
  - runtime: 127s
  - verifier used: true
  - selected candidate role: `sponsor_or_advertiser`
  - score: 13
  - TD 来自 relation extraction
  - `peaksaver` 被判为 `article_subject / related_but_wrong_role`
  - `Brock News` 被判为 `source_or_publication`，不允许 final
  - subagent max_calls: 0
  - DSML/tool-call/JSON-like final 污染: 0

## 6. 当前效果判断

当前 patch 已经形成较清晰的 harness 优化链条：

- v1 降低 sub-agent max_calls；
- v2 建立 candidate ledger 和 final verifier；
- v2 conservative 避免 verifier 改坏正确样本；
- v2.1 relation-aware scoring 在 item 18 上完成 targeted correction。

但目前只做了 targeted validation，尚未跑完整 patch 后 BrowseComp 20 条，因此不能宣称 benchmark-level improvement。可以诚实表述为：当前 harness 在特定失败机制上已有可观测改善。

## 7. 下一步建议

1. 只跑 item 15，验证 `Phaistos Disc` 是否能被识别为 clue_object，并尝试修正到 `The Genius of the Few`；
2. 如果 item 15 成功或部分成功，再跑 item 13；
3. item 14 主要用于 runtime / early finalize 验证，放后面；
4. 最后才考虑完整 20 条 patched Swarm 对比；
5. 若 API 成本有限，可保持 targeted validation，不强行跑完整 benchmark。
