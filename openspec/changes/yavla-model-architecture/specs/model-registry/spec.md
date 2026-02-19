## ADDED Requirements

### Requirement: Generic Registry class
`Registry[ConfigT, ModuleT]` SHALL map string names to `(config_class, module_class)` pairs, with `register(name)` decorator, `build(config) → ModuleT`, `list() → list[str]`, and `get_default_config(name) → ConfigT`.

#### Scenario: Register and build
- **WHEN** `@registry.register("flow_matching")` decorates a class and `registry.build(config)` is called with `config.type == "flow_matching"`
- **THEN** it SHALL instantiate and return the registered class

#### Scenario: Entry-point plugin discovery
- **WHEN** `registry.discover_plugins(group="yavla.action_heads")` is called
- **THEN** it SHALL load entry points from installed packages and register them

#### Scenario: Duplicate registration error
- **WHEN** two classes register with the same name
- **THEN** `ValueError` SHALL be raised identifying the duplicate

#### Scenario: Unknown module error
- **WHEN** `registry.build(config)` is called with `type="nonexistent"`
- **THEN** `KeyError` SHALL be raised listing available names
