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
In simple terms: Pricing a fixed income instrument, like a swap, requires yield curves to act as forecast of rates and/or discount factors. Yield curves are calibrated to market quotes of fixed income instruments (as per some config per curveset). Pricing is going from curve -> quotes, calibrating is going from quotes -> curves.

By using algorithmic differentiation (AD), we can only write one of the paths, and let AD take care of the reverse. That is, we can just define how to price a swap off some curve(s), and let AD give us the corresponding jacobian from swap quotes to curves.

This idea is certainly not novel. Algorithmic differentiation has been around for more than half a century and is widely used in a number of fields. ML in particular has resulted in the emergence of powerful libraries like Pytorch, TensorFlow and JAX, which have made python the reigning ground for ML model development. Finance has also adopted similar approaches, as that allows, for example, the very efficient computation of the effects of different risk factors for large portfolios with adjoint AD (AAD).

Coming back to the problem of curve calibration with AD, a simple example appears in J. H. M. Darbyshire, Pricing and Trading Interest Rate Derivatives: A Practical Guide to Swaps by J. H. M. Darbyshire (3rd edition, 2022). In it, he explains and uses the AD approach of dual numbers to exemplify how to calibrate a curve from swap quotes.

In this toy project, I want to explore that same core idea of how a fixed income quant library would use AD for both pricing and calibration, but using the JAX library. [JAX](https://docs.jax.dev/en/latest/) is a powerful library widely used in ML circles with a numpy-style api and extensive tooling. Particularly interesting is that it supports JIT compilation for lightning-fast computation, while only developing at the python level. One can write relatively straightforward pricing functions, and let jax handle the optimisation and dispatch of jacobians for a whole curveset. However, JAX also enforces a number of constraints. First, all code that leverages JAX needs to use JAX natives (JAX ndarrays, etc.) and JAX-compatible code (e.g., no if conditions); this means that either the whole library only speaks jax, or any kind of public/user api should handle converting inputs to JAX compatibles, and sit on top of a JAX kernel. Second, JAX uses functional programming, which constraints OOP designs, dynamic dispatches, etc. Third, jitted functions are cached by input shapes; using the same function to price, say, 10 swaps first and then 100 swaps would require a new jit compilation the second time around, with the corresponding overhead.

These constraints drive the design of one such library, and the purpose of this personal package is both to explore that in its minimal form, as well as inspecting the corresponding performance. A POC script can be found in _poc/poc.py, which calibrates a simple curve (in instantaneous forward rates space [HJM]) with 30 input instruments in less than 10ms, with a jacobian evaluation cost of only 0.2ms. When tested with 500 instruments, calibration remains below 100ms. This is good performance for the core calibration engine. Naturally, in a python production library, there would be  more overhead from tooling not related to the core calibration (tooling around conventions, dates, market data, etc.); or one might have a C++ library that uses AD directly without the need of JIT. But as a general idea for the plumbing behind a python quant library, I found it an interesting approach.

# Pros & Cons
### Advantages
- Same code does pricing and calibration. Ensures consistency by design and saves developer time (quants are expensive!).
- Compiled performance with python code.

### Disadvantages
- Design must work around JAX constraints (functional-first, difficult for OOP, no if conditions, etc.)
- At least the kernel must be pure jax. If required to speak another language (e.g. numpy), there should be a translation layer at the public api level.
- To prevent JIT warm-up overhead, function inputs must remain the same shape as much as possible. May require e.g. padding.



