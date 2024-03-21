# Tutorial -- GPU accelerated computing

## Motivation

**Utilize the additional speed provided by compilation:**

Python is an interpreted language, where each line is executed one by one by the interpreter and convert to python bytecode.
- Conversion incurs a time cost, which occurs when the code is actually ran
- Time cost occurs even when same instructions are run
- No look ahead after current line, so no optimizations possible by "combining multiple lines"


**Utilize the GPU, which excels at parallel tasks**:

-Many computing jobs do not require sequential execution e.g. vector addition
-Machine code executes sequentially on CPUs (when no parallelization), must be done one by one.
-If sequentiality is not required, GPU can execute instructions faster because there are more threads availiable to do work

## Code tutorial

**"Free Lunch" techniques**: [see notebook](<free lunch.ipynb>) (WIP)

Usage of decorators to invoke compilation (WIP)
- jit (Done)
- vectorize (WIP)

Drop in replacement libraries for instant speedup (TBD)

**CUDA python tutorial (with Numba)**: [see notebook](<numba cuda kernel.ipynb>) (TBD)

Create kernels (an operation on input arraylike)

Accessing data in "correct" order for efficient memory access
