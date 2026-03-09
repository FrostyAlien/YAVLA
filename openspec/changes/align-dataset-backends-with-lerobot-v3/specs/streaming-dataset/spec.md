## MODIFIED Requirements

### Requirement: Streaming backend is experimental
YAVLA SHALL treat the streaming backend as experimental rather than as a primary supported training backend.

#### Scenario: Public streaming guidance is explicit
- **WHEN** docs, guides, or capability descriptions mention the streaming backend
- **THEN** they SHALL label it as experimental

#### Scenario: Random-access workloads are redirected away from streaming
- **WHEN** docs discuss workloads that need true random access, sampler-driven uniform shuffling, or stable distributed epoch behavior
- **THEN** they SHALL direct users to `default` or `lazy` rather than to `streaming`

### Requirement: Streaming replacement is deferred to backlog
Replacing the current experimental streaming backend with upstream `StreamingLeRobotDataset` SHALL be treated as backlog follow-up work rather than as an in-scope requirement of this change.

#### Scenario: Future replacement is framed as follow-up work
- **WHEN** change-local docs or notes discuss upstream `StreamingLeRobotDataset`
- **THEN** they MAY identify it as the likely future replacement direction, but SHALL NOT describe that replacement as already implemented or required by this change

#### Scenario: Existing implementation semantics remain current
- **WHEN** this change is applied without a follow-up streaming replacement change
- **THEN** the current streaming implementation contract SHALL remain the active implementation until a later change explicitly replaces or retires it
