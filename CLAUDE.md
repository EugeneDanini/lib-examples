# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Python learning/reference repository containing examples and demonstrations of:
- **Python library usage** (memory-profiler, pympler, objgraph, google-adk)
- **SOLID principles** (SRP, OCP, LSP, ISP, DIP)
- **Design patterns** (Circuit Breaker, Observer, Singleton, Interceptor, Retry Storm, Throttle, Dependency Injection, Single Point of Failure)
- **Distributed systems patterns** (Robustness, Scalability, Performance, Availability, Extensibility, Resiliency, CAP Theorem)
- **Non-functional requirements** (Performance, Security, Reliability, Scalability, Maintainability, Usability, Observability)
- **System design frameworks** (RESHADED approach)
- **SQL isolation levels** (READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE)

## Project Structure

```
lib-examples/
├── lib_<library_name>/          # Library-specific examples
│   ├── __init__.py
│   ├── example.py               # Runnable demonstration
│   └── readme.md (optional)     # Library-specific notes
├── ai_examples/                 # Educational/reference code
│   ├── distributed_systems/     # Distributed system patterns
│   │   ├── robustness.py
│   │   ├── scalability.py
│   │   ├── performance.py
│   │   ├── availability.py
│   │   ├── extensibility.py
│   │   ├── resiliency.py
│   │   └── cap_theorem.py
│   ├── solid/                   # SOLID principles implementations
│   ├── patterns/                # Design pattern implementations
│   ├── non_functional_requirements.py
│   ├── reshaded.md              # System design framework guide
│   └── isolation.sql            # SQL transaction isolation examples
├── requirements.txt             # All dependencies
└── new.sh                       # Script to add new library examples
```

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Examples

**Standard library examples:**
```bash
python lib_<library_name>/example.py
```

**Memory Profiler specific:**
```bash
mprof run lib_memory_profiler/example.py
mprof plot
```

**ObjGraph (generates PNG output):**
```bash
python lib_objgraph/example.py
# Creates example.png visualizing object references
```

**AI Examples (educational/reference code):**
```bash
# SOLID principles
python ai_examples/solid/<principle>.py

# Design patterns
python ai_examples/patterns/<pattern>.py

# Distributed systems patterns
python ai_examples/distributed_systems/<pattern>.py

# Non-functional requirements
python ai_examples/non_functional_requirements.py
```

### Adding New Library Examples

Use the `new.sh` script to set up a new library example:
```bash
./new.sh <library-name>
```

This script:
1. Uninstalls all current packages
2. Reinstalls from requirements.txt
3. Installs the specified library
4. Updates requirements.txt with the new frozen dependencies
5. Creates `lib_<library_name>/` directory with `__init__.py` and `example.py`

## Architecture Notes

### Library Examples Pattern
Each `lib_*/example.py` follows a similar structure:
- Import the library being demonstrated
- Define `EXAMPLE_OBJECT` (often a simple dict for testing)
- Implement a `run()` function containing the demonstration
- Include `if __name__ == '__main__'` guard

**Existing library examples:**
- `lib_memory_profiler/`: Line-by-line memory usage profiling with `mprof` CLI tool
- `lib_pympler/`: Memory tracking with `asizeof`, `classtracker`, and `tracker` modules
- `lib_objgraph/`: Object reference visualization (outputs PNG files with graphviz)
- `lib_google_adk/`: Google ADK (Agent Development Kit) demonstrations

### AI Examples Directory
Contains **reference implementations** demonstrating software engineering concepts. All files are runnable and self-contained:

- **solid/**: SOLID principles (SRP, OCP, LSP, ISP, DIP) with concrete examples
- **patterns/**: Design patterns (Observer, Singleton, Circuit Breaker, Dependency Injection, Single Point of Failure, Retry Storm, Throttle, Interceptor) with runnable demonstrations
- **distributed_systems/**: Comprehensive distributed systems patterns
  - Each file demonstrates a key aspect: robustness, scalability, performance, availability, extensibility, resiliency
  - CAP theorem with CP, AP, and Quorum system implementations
  - All examples include demonstrations that can be executed directly
- **non_functional_requirements.py**: Seven categories of NFRs with measurable, testable implementations
- **reshaded.md**: Documentation of the RESHADED system design framework (Requirements, Estimation, Storage, High-level design, API design, Detailed design, Evaluation, Deep-dive)
- **isolation.sql**: SQL transaction isolation level demonstrations (READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE)

Each Python file in ai_examples/ follows this pattern:
- Comprehensive docstring explaining the concept
- Multiple classes demonstrating different approaches
- `demonstrate_*()` functions showing practical usage
- `main()` function that runs all demonstrations
- Direct execution via `if __name__ == '__main__'`

### Dependencies
The repository uses a comprehensive requirements.txt with many libraries. When working with a specific example, only the relevant library's imports are used in that module.

### Naming Convention
The repository has migrated from `Docs/` to `ai_examples/` for educational content. When referencing or creating new examples, use the `ai_examples/` directory structure.

**Note**: The throttle pattern file is named `throtlle.py` (with a typo) - reference it with this spelling when running or importing it.