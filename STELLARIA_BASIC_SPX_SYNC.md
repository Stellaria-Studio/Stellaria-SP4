# Stellaria BasicX SPX Consolidation Note

Stellaria BasicX now owns the SP4 native backend surface under
`runtime/src/RHI/Extensions/Stellaria SPX/Stellaria SP4` on `main`.

The Basic-side copy is restricted to C++ SDK code, Slang/Metal shaders and
documentation. Python toolchain prototypes, pycache, build output and nested
repository metadata are intentionally excluded from the RHI extension path.

This standalone repository is frozen after the migration and is not a source
for future synchronization. Integration ownership:

- Stellaria BasicX owns macro scheduling, frame ownership, resource routing,
  queues and backend handoff.
- Compute Graph Extension owns compute-only RHI task descriptors.
- SPX/SP4 owns native micro scheduling for manually dispatched complex compute
  tasks.
- SP backend construction, compiler decisions and scheduler decisions must be
  native C++/Slang/Metal in production paths, not Python.
