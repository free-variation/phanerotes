# Project Guidelines for Claude

## Communication Style

- Don't use validating phrases like "you're right", "fair point", "good question"
- State positions directly without preamble
- When asked to take a stand, take one

## Building and Running

To run an example program:
```bash
fpm run --example <program_name> --profile release --flag "-march=native -ffast-math -fopenmp"
```

To run an example with arguments:
```bash
fpm run --example <program_name> --profile release --flag "-march=native -ffast-math -fopenmp" -- <args>
```

## Memory Layout

Tensor layout is `(channels, height, width, batch)` - Fortran column-major order.

im2col orders columns with width varying fastest (row-major spatial). Use `order=[1,3,2]` in reshape to match this when converting between matrix and tensor forms.
