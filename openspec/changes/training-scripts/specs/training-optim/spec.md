## ADDED Requirements

### Requirement: make_optimizer_and_scheduler factory
`make_optimizer_and_scheduler(policy, config: TrainingConfig, num_training_steps: int) -> tuple[Optimizer, LRScheduler]` SHALL be implemented in `src/yavla/training/optim.py`.

It SHALL:
1. If `config.use_policy_preset` is `True` and `policy.get_optimizer_preset()` returns non-None, merge preset fields over `config.optimizer` defaults
2. Build two param groups: backbone params (`policy.backbone.parameters()`) at `lr * backbone_lr_scale`, all other params at full `lr`
3. Construct `torch.optim.AdamW` (or the optimizer named in `config.optimizer.name`) with the two param groups
4. Build `SequentialLR([LinearLR(start_factor=0.01, total_iters=warmup_steps), CosineAnnealingLR(T_max=remaining_steps, eta_min=lr*min_lr_ratio)], milestones=[warmup_steps])` using built-in PyTorch schedulers
5. Return `(optimizer, scheduler)`

The returned optimizer and scheduler are subsequently wrapped by `accelerator.prepare()` in the `Trainer`.

#### Scenario: Warmup phase
- **WHEN** the scheduler is stepped for `warmup_steps` steps
- **THEN** the LR SHALL increase monotonically from near-zero to `config.optimizer.lr`

#### Scenario: Cosine decay phase
- **WHEN** the scheduler is stepped past `warmup_steps`
- **THEN** the LR SHALL decrease following a cosine curve toward `lr * min_lr_ratio`

#### Scenario: Policy preset override
- **WHEN** `use_policy_preset=True` and `policy.get_optimizer_preset()` returns `OptimizerConfig(lr=5e-5)`
- **THEN** the optimizer SHALL use `lr=5e-5` regardless of `config.optimizer.lr`

#### Scenario: Backbone param group
- **WHEN** the optimizer is constructed
- **THEN** `optimizer.param_groups` SHALL have 2 groups: backbone at `lr * backbone_lr_scale`, rest at full `lr`
