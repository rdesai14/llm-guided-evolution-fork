# Training block: optimizer and train_angles contract

`main()` trains every circuit 10 times. Each time it draws different starting angles
and calls `train_angles(qc, x_params, w_params, X, y, w0)`, which must return a 1-D
numpy array of `len(w_params)` angles. Keep that name and signature exactly. Start the
search from the `w0` you are given; do not draw your own starting angles or set a seed.
A fixed start (for example `default_rng(42)` inside `train_angles`) makes all 10
trainings start from the same angles, and such a circuit is rejected.
Inside, compute the loss with `cross_entropy(qc, x_params, w_params, w, X, y)`; the
protected training budget (178 passes over the training set) stops the search by
raising `BudgetSpent`, which `main()` catches - do not swallow it inside your own loop.
Valid scipy optimizers (`minimize` is already imported): `minimize(objective, w0,
method="COBYLA" | "Powell" | "Nelder-Mead" | "L-BFGS-B", options={"maxiter": MAX_ITER})`.
Angles are periodic, so wrap with `np.mod(w, 2 * np.pi)` rather than clipping.
