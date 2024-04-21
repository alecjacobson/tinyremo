# 🥁 TinyReMo 🥁

Tiny header-only, minimal dependency reverse-mode automatic differentiation library for C++.

Based on the tape-based implmenetation tutorial at https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation

## Compile and Run Tests

```bash 
clang++ -std=c++20 -I . -I [path/to/eigen] -I -o test test.cpp
./test
```
