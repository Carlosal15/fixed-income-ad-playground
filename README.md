# fixed-income-ad-playground

<!-- [![Release](https://img.shields.io/github/v/release/carlosal15/fixed-income-ad-playground)](https://img.shields.io/github/v/release/carlosal15/fixed-income-ad-playground)
[![Build status](https://img.shields.io/github/actions/workflow/status/carlosal15/fixed-income-ad-playground/main.yml?branch=main)](https://github.com/carlosal15/fixed-income-ad-playground/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/carlosal15/fixed-income-ad-playground/branch/main/graph/badge.svg)](https://codecov.io/gh/carlosal15/fixed-income-ad-playground)
[![Commit activity](https://img.shields.io/github/commit-activity/m/carlosal15/fixed-income-ad-playground)](https://img.shields.io/github/commit-activity/m/carlosal15/fixed-income-ad-playground)
[![License](https://img.shields.io/github/license/carlosal15/fixed-income-ad-playground)](https://img.shields.io/github/license/carlosal15/fixed-income-ad-playground) -->

This is a toy/playground for a fixed income library that uses algorithmic differentiation (via jax) for curve calibration and pricing.

- **Github repository**: <https://github.com/carlosal15/fixed-income-ad-playground/>
<!-- - **Documentation** <https://carlosal15.github.io/fixed-income-ad-playground/> -->

# Motivation

In simple terms: pricing a fixed income instrument, like a swap, requires yield curves to act as forecasts of rates and/or discount factors. Yield curves themselves are calibrated to market quotes of fixed income instruments (as per some curveset config). Pricing is going from curve → quotes; calibration is going from quotes → curves.

By using algorithmic differentiation (AD), we can write only one of these paths and let AD “take care” of the reverse (i.e. the derivatives). That is, we define how to price a swap off some curve(s), and let AD provide the corresponding Jacobians of instrument prices with respect to curve parameters.

This idea is certainly not novel. Algorithmic differentiation has been around for more than half a century and is widely used in a number of fields. Machine learning in particular has resulted in the emergence of powerful libraries like PyTorch, TensorFlow, and JAX, which have made Python the dominant language for ML model development. Finance has also adopted similar approaches, for example to efficiently compute sensitivities of large portfolios via adjoint algorithmic differentiation (AAD).

Coming back to the problem of curve calibration with AD, a simple example appears in *Pricing and Trading Interest Rate Derivatives: A Practical Guide to Swaps* by J. H. M. Darbyshire (3rd edition, 2022). In it, he uses a forward-mode AD approach based on dual numbers to illustrate how one can calibrate a curve directly from swap quotes.

In this project, I explore the same core idea of how a fixed income quant library could use AD for both pricing and calibration, but using the JAX library. [JAX](https://docs.jax.dev/en/latest/) is a powerful library widely used in ML circles, with a NumPy-style API and extensive tooling. Particularly interesting is that it supports JIT compilation for lightning-fast computation while developing entirely at the Python level.

One can write relatively straightforward pricing functions and let JAX handle the computation and dispatch of Jacobians for an entire curveset. However, JAX also enforces a number of constraints. First, all code that runs inside JAX-compiled kernels must use JAX primitives (JAX arrays, `jax.numpy`, etc.) and avoid Python control flow that depends on traced values (e.g. standard `if` statements inside jitted code). This would mean that either the whole library must speak JAX, or that any public/user-facing API must translate inputs into JAX-compatible representations and sit on top of that JAX kernel. Second, JAX follows a functional programming model, which constrains OOP designs, dynamic dispatch, and inheritance-heavy architectures. Third, JIT-compiled functions are cached by input shapes; using the same function to price, say, 10 swaps first and then 100 swaps would require a new JIT compilation the second time around, with the corresponding overhead.

These constraints drive the design of such a library. The purpose of this toy project is to explore this design space in a minimal setting, as well as to see what performance it could achieve. A POC script can be found in `_poc/poc.py`, which calibrates a simple curve (in instantaneous forward rate space, HJM-style) with 30 input instruments in less than 10ms, with a Jacobian evaluation cost of around 0.2ms. When tested with 500 instruments, calibration remains below 100ms.

This is good performance for the core calibration engine. Naturally, a production Python library would incur additional overhead from tooling unrelated to the core calibration logic (conventions, dates, market data, orchestration, etc.), or one might use a C++ library that applies AD directly without relying on JIT compilation. But as a general idea for the plumbing behind a python quant library, I found it an interesting approach.

# Pros & Cons

### Advantages
- The same code is used for pricing and calibration, ensuring consistency by design and reducing development work (quants are expensive!).
- Compiled performance while writing Python code.

### Disadvantages
- The design must work around JAX constraints (functional-first style, limited OOP patterns, restricted control flow inside JIT).
- At least the kernel must be pure JAX; if another interface is required (e.g. NumPy), a translation layer is needed at the public API level.
- To prevent JIT warm-up overhead, function inputs should remain shape-stable as much as possible, which may require padding or packing strategies.

# Goals and Non-goals

- Explore a design that uses JAX at the center of curve calibration, both in terms of implementation and achievable performance.
- Provide a demo public API for defining configurations, instrument conventions, requesting pricing and metrics, etc.

A fully fledged library would require significantly more plumbing (additional instrument types and derivatives, calendars and date logic, conventions, risk metrics, volatility models, and a large etc.). At the time of writing, this repository contains only a proof-of-concept with a single step-wise constant curve and mock OIS swaps. The first goal is to refactor and extend this into a more library-like structure with clearer interfaces and encapsulation, which can then be expanded further.

**Non-goal**: building a production-ready fixed income library. This is an exploratory project focused on a core design idea.

# Who this is for
- Me.
- Anyone implementing curve calibration in Python who wants to explore what performance and design trade-offs are possible with AD and JAX.

<!-- # Architectural overview
To be filled in later. -->