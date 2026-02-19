## ADDED Requirements

### Requirement: Generic Registry class
`Registry[ConfigT, ModuleT]` SHALL be a generic class that maps string names to `(config_class, module_class)` pairs, with `register(name)` decorator, `build(config: ConfigT) → ModuleT`, `list() → list[str]`, and `get_default_config(name)`.

#### Scenario: Register and build a module
- **WHEN** a class is decorated with `@registry.register("my_module")` and `registry.build(config)` is called with a config whose `type` field matches `"my_module"`
- **THEN** the registry SHALL instantiate and return the registered class

#### Scenario: List registered modules
- **WHEN** `registry.list()` is called after registering modules `"a"` and `"b"`
- **THEN** it SHALL return `["a", "b"]` (or equivalent containing both names)

#### Scenario: Duplicate registration error
- **WHEN** two classes are registered with the same name
- **THEN** the registry SHALL raise `ValueError` with a message identifying the duplicate name

#### Scenario: Unknown module build error
- **WHEN** `registry.build(config)` is called with `type="nonexistent"`
- **THEN** the registry SHALL raise `KeyError` listing available module names
