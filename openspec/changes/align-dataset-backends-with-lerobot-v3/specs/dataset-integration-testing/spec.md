## MODIFIED Requirements

### Requirement: Integration test scope reflects supported backend tiers
Integration testing for this change SHALL continue to prioritize the supported random-access training paths, while experimental streaming replacement work remains out of scope.

#### Scenario: Default and lazy remain the primary compatibility targets
- **WHEN** real-data integration coverage is planned for this change
- **THEN** `default` and `lazy` SHALL remain the primary backends whose behavior is treated as part of the supported training contract

#### Scenario: Upstream streaming parity coverage is deferred
- **WHEN** deciding whether to add new integration tests for upstream `StreamingLeRobotDataset`
- **THEN** that coverage SHALL be treated as backlog follow-up work, not as a required part of this change
