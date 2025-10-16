# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Python learning/reference repository containing examples and demonstrations of:
- **Python library usage** (memory profiling, object graph visualization, memory analysis, Google ADK)
- **SOLID principles** (SRP, OCP, LSP, ISP, DIP)
- **Design patterns** (Circuit Breaker, Observer, Singleton, Interceptor, Retry Storm, Throttle)
- **SQL isolation levels** (READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE)

## Project Structure

```
lib-examples/
├── lib_<library_name>/          # Library-specific examples
│   ├── __init__.py
│   ├── example.py               # Runnable demonstration
│   └── readme.md (optional)     # Library-specific notes
├── Docs/                        # Educational/reference code
│   ├── solid/                   # SOLID principles implementations
│   ├── patterns/                # Design pattern implementations
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

**SOLID/Pattern examples:**
```bash
python Docs/solid/<principle>.py
python Docs/patterns/<pattern>.py
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

### Docs Directory
Contains **reference implementations** only—not runnable applications. These demonstrate software engineering concepts (SOLID, design patterns) and should be kept simple and focused on the principle being illustrated.

### Dependencies
The repository uses a comprehensive requirements.txt with many libraries. When working with a specific example, only the relevant library's imports are used in that module.