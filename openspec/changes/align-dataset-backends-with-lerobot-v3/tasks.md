## 1. Narrow The Change Scope

- [ ] 1.1 Update change-local design and spec language so `streaming` is described as experimental rather than as a peer supported training backend
- [ ] 1.2 Record possible replacement of the experimental streaming backend with upstream `StreamingLeRobotDataset` as backlog follow-up work instead of in-scope implementation work

## 2. Plan Documentation Updates

- [ ] 2.1 Update dataset-layer docs to describe `default` as the upstream-native baseline and `lazy` as the supported random-access backend for larger datasets
- [ ] 2.2 Mark `backend="streaming"` as experimental in the backend guide, architecture docs, caveats, and training guide
- [ ] 2.3 Document that streaming is not the recommended backend for workloads that require true random access, strong sampler semantics, or stable distributed epoch behavior
- [ ] 2.4 Note in docs that the experimental streaming backend may be replaced by upstream `StreamingLeRobotDataset` in a future change, without promising identical semantics today

## 3. Backlog Follow-Up

- [ ] 3.1 If a real streaming use case appears, evaluate upstream `StreamingLeRobotDataset` against YAVLA needs for action windows, distributed iteration, epoch control, and decoder behavior before changing the backend implementation
