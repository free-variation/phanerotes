# Project Guidelines for Claude

## Communication Style

- Don't use validating phrases like "you're right", "fair point", "good question"
- State positions directly without preamble
- When asked to take a stand, take one

## Code Style

- Always seek elegant, economical solutions
- Avoid boilerplate; leverage libraries and existing utilities
- If something feels verbose, there's probably a better way

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

## Testing

Tests go in `/test/`, not in `/example/`. Uses test-drive framework:
- Test modules named `test_<module>.f90`
- Public `collect_<module>_tests(testsuite)` subroutine
- Each test is a subroutine taking `error_type` output
- Use `check(error, condition, message)` for assertions

Run tests with:
```bash
fpm test --profile release --flag "-march=native -ffast-math -fopenmp"
```
