"""World-state compiler: screenplay (+ current world state) -> bounded shot requests.

The recursive compiler exposes three calls used by the CLI and the execution loop:

    initial_plan()           the plan known before any generation (may be partial)
    requests(runner, limit)  yields bounded requests in order; it reads
                             ``runner.world`` between yields and extends ``runner.plan``
    compile_all()            plan everything without generating media (inspection)
"""
from atas.compiler.recursive import RecursiveCompiler
__all__ = ["RecursiveCompiler"]
